# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Models used for inference."""

import abc
import os
import enum
from typing import Any, List
from argparse import Namespace
import transformers
import torch

from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig
from transformers.cache_utils import OffloadedCache

from absl import logging
from inference import utils
import vertexai
from vertexai.generative_models import GenerationConfig
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import HarmCategory
from vertexai.generative_models import Part
from vertexai.generative_models import SafetySetting


ContentChunk = utils.ContentChunk
MimeType = utils.MimeType
LOCATION = 'us-central1'
TEMPERATURE = 0.0


class GeminiModel(enum.StrEnum):
    GEMINI_1_5_FLASH_002 = 'gemini-1.5-flash-002'  # Max input tokens: 1,048,576
    GEMINI_1_5_PRO_002 = 'gemini-1.5-pro-002'  # Max input tokens: 2,097,152


class LlamaModel(enum.StrEnum):
    LLAMA_3_2_1B_INSTRUCT = "llama3.2-1b-instruct"


class SupportedModels(enum.StrEnum):
    GEMINI_1_5_FLASH_002 = 'gemini-1.5-flash-002'  # Max input tokens: 1,048,576
    GEMINI_1_5_PRO_002 = 'gemini-1.5-pro-002'  # Max input tokens: 2,097,152
    LLAMA_3_2_1B_INSTRUCT = "llama3.2-1b-instruct"


class Model(metaclass=abc.ABCMeta):
    """Base class for models."""

    def index(
        self,
        content_chunks: List[ContentChunk],
        document_indices: List[tuple[int, int]],
        **kwargs: Any,
    ) -> str:
        """Indexes the example containing the corpus.

        Arguments:
          content_chunks: list of content chunks to send to the model.
          document_indices: list of (start, end) indices marking the documents
            boundaries within content_chunks.
          **kwargs: additional arguments to pass.

        Returns:
          Indexing result.
        """
        del content_chunks, document_indices, kwargs  # Unused.
        return 'Indexing skipped since not supported by model.'

    @abc.abstractmethod
    def infer(
        self,
        content_chunks: List[ContentChunk],
        document_indices: List[tuple[int, int]],
        **kwargs: Any,
    ) -> str:
        """Runs inference on model and returns text response.

        Arguments:
          content_chunks: list of content chunks to send to the model.
          document_indices: list of (start, end) indices marking the documents
            boundaries within content_chunks.
          **kwargs: additional arguments to pass to the model.

        Returns:
          Inference result.
        """
        raise NotImplementedError


class HFAiModel(Model):
    def __init__(
        self,
        pid_mapper,
        model,
        tokenizer,
        device,
        args,
    ):
        self.pid_mapper = pid_mapper
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.args = args

    def infer(self, content_chunks: List[ContentChunk], **kwargs):

        lines = list(map(lambda x: x._text, content_chunks))
        prompt = "\n".join(lines)

        # print(f"{prompt=}")

        messages = [
            {"role": "user", "content": prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors='pt',
            truncation=False,
        )
        seq_len = inputs.shape[-1]

        additional_args = {}
        past_key_values = OffloadedCache()

        with torch.no_grad():
            inputs = inputs.cuda()

            print(f"after putting inputs on cuda: {seq_len=}")
            _output = self.model.generate(
                inputs=inputs,
                past_key_values=past_key_values,
                max_new_tokens=self.args.max_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **additional_args,
            )
            output: str = self.tokenizer.decode(
                _output[0][seq_len:].data.cpu(),
                skip_special_tokens=True,
            )

        print(output)

        if output.endswith('</s>'):
            output = output[:-4]
        text = output.strip()

        text = text.replace('[|endofturn|]', '')
        text = text.replace('<|eot_id|>', '')
        text = text.replace('<end_of_turn>', '')
        print('Generated:', text.replace('\n', '\\n'))

        final_answers = utils.extract_prediction(text)
        print(f"after extract prediction: {final_answers}")
        if os.getenv('IGNORE_PID_MAPPER', '0') == '0':
            final_answers = [
                self.pid_mapper[str(answer)] if str(answer) in self.pid_mapper else str(answer) for answer in final_answers
            ]
        else:
            final_answers = [text]

        print('Final Answer:', final_answers, flush=True)

        return final_answers


class VertexAIModel(Model):
    """GCP VertexAI wrapper for general Gemini models."""

    def __init__(
        self,
        project_id: str,
        model_name: str,
        pid_mapper: dict[str, str],
    ):
        self.project_id = project_id
        self.model_name = model_name
        self.pid_mapper = pid_mapper
        vertexai.init(project=project_id, location=LOCATION)
        self.model = GenerativeModel(self.model_name)

    def _process_content_chunk(self, content_chunk: ContentChunk) -> Part:
        if content_chunk.mime_type in [
            MimeType.TEXT,
            MimeType.IMAGE_JPEG,
            MimeType.AUDIO_WAV,
        ]:
            return Part.from_data(
                content_chunk.data, mime_type=content_chunk.mime_type
            )
        else:
            raise ValueError(f'Unsupported MimeType: {content_chunk.mime_type}')

    def _get_safety_settings(
        self, content_chunks: List[ContentChunk]
    ) -> List[SafetySetting]:
        """Returns safety settings for the given content chunks."""
        # Audio prompts cannot use BLOCK_NONE.
        if any(
            content_chunk.mime_type == MimeType.AUDIO_WAV
            for content_chunk in content_chunks
        ):
            threshold = SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
        else:
            threshold = SafetySetting.HarmBlockThreshold.BLOCK_NONE
        return [
            SafetySetting(
                category=category,
                threshold=threshold,
            )
            for category in HarmCategory
        ]

    def _postprocess_response(self, response: Any) -> List[str]:
        """Postprocesses the response from the model."""
        try:
            output_text = getattr(response, 'candidates')[0].content.parts[0].text
            final_answers = utils.extract_prediction(output_text)
            final_answers = [
                self.pid_mapper[str(answer)] for answer in final_answers
            ]
        except Exception as e:  # pylint:disable=broad-exception-caught
            logging.error('Bad response %s with error: %s', response, str(e))
            raise ValueError(f'Unexpected response: {response}') from e

        return final_answers

    def infer(
        self,
        content_chunks: List[ContentChunk],
        **kwargs: Any,
    ) -> List[str]:
        response = self.model.generate_content(
            [
                self._process_content_chunk(content_chunk)
                for content_chunk in content_chunks
            ],
            generation_config=GenerationConfig(temperature=TEMPERATURE, top_p=1.0),
            safety_settings=self._get_safety_settings(content_chunks),
        )

        return self._postprocess_response(response)


def get_model(
    model_url_or_name: str,
    project_id: str | None,
    pid_mapper: dict[str, str],
) -> Model:
    """Returns the model to use."""

    if model_url_or_name in GeminiModel.__members__.values():
        if project_id is None:
            raise ValueError(
                'Project ID and service account are required for VertexAIModel.'
            )
        model = VertexAIModel(
            project_id=project_id,
            model_name=model_url_or_name,
            pid_mapper=pid_mapper,
        )
    elif model_url_or_name in LlamaModel.__members__.values():
        PATH = "/d1/dataset/llama/models/llama_v3.1/"
        MODELS = {
            'llama3.2-1b-instruct': 'meta-llama/Llama-3.2-1B-Instruct',
            'llama3.2-1b': 'meta-llama/Llama-3.2-1B',
            'llama3.1-8b-instruct': os.path.join(PATH, "Meta-Llama-3.1-8B-Instruct"),
            'llama3.1-8b': os.path.join(PATH, "Meta-Llama-3.1-8B"),
            'llama3.1-70b': os.path.join(PATH, "Meta-Llama-3.1-70B"),
            'llama3.1-70b-instruct': os.path.join(PATH, "Meta-Llama-3.1-70B-Instruct"),
            'llama3.1-70b-instruct-gptq-int4': "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4",
            'llama7b': 'togethercomputer/LLaMA-2-7B-32K',
            'llama13b': 'meta-llama/Llama-2-13b-hf',
            'llama13b_32k': 'Yukang/Llama-2-13b-longlora-32k-ft',
            'llama7b-chat': '/d1/dataset/llama/models/llama_v2/llama-2-7b-chat-hf',
            "llama2-7b-chat-32k": "togethercomputer/Llama-2-7B-32K-Instruct",
            'qwen14b': 'Qwen/Qwen1.5-14B',
            'qwen7b': 'Qwen/Qwen1.5-7B',
            'qwen7b-chat': 'Qwen/Qwen1.5-7B-Chat',
            "qwen2-14b-chat-32k": "Qwen/Qwen1.5-14B-Chat",
            "qwen2-7b-chat-32k": "Qwen/Qwen1.5-7B-Chat",
            "qwen2-7b-instruct": "Qwen/Qwen2-7B-Instruct",
            "qwen2-7b": "Qwen/Qwen2-7B",
            'qwen0.5b': 'Qwen/Qwen1.5-0.5B',
            'llama1.3b': 'princeton-nlp/Sheared-LLaMA-1.3B',
            'llama3-8b-instruct':
            "/d1/dataset/llama/models/llama_v3/Meta-Llama-3-8B-Instruct",
            'llama3-8b': 'meta-llama/Meta-Llama-3-8B',
            'llama3-70b-instruct':
            "/d1/dataset/llama/models/llama_v3/Meta-Llama-3-70B-Instruct",
            'llama2-70b': "/d1/dataset/llama/models/llama_v2/llama-2-70b",
        }

        device = 'cuda:0'

        args = Namespace(
            device=device,
            model=model_url_or_name,
            local_rank=int(os.getenv('LOCAL_RANK', '0')),
            world_size=int(os.getenv('WORLD_SIZE', '1')),
            batch_size=1,
            window=int(os.getenv('WINDOW', '0')),
            method=str(os.getenv("METHOD", "plain")),
            max_tokens=1024,
        )

        assert args.model in MODELS, MODELS.keys()
        model_id = MODELS[args.model]

        config = LlamaConfig.from_pretrained(model_id)
        config._attn_implementation = config.attn_implementation = 'flash_attention_2'

        config._batch_size = args.batch_size
        config._window = args.window
        config.world_size = args.world_size
        config._method = args.method

        print(f"{config=}")

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

        args.infer_dtype = torch.float16
        from_pretrained_kwargs = dict(
            config=config,
            device_map={"": device},
            torch_dtype=args.infer_dtype,
        )

        _model = LlamaForCausalLM.from_pretrained(model_id, **from_pretrained_kwargs)
        model = HFAiModel(
            pid_mapper=pid_mapper,
            model=_model,
            tokenizer=tokenizer,
            device=device,
            args=args,
        )

        return model

    else:
        raise ValueError(f'Unsupported model: {model_url_or_name}')
    return model
