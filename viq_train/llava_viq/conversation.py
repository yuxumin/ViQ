import dataclasses
from enum import auto, Enum
from typing import List, Tuple
import re
import base64
from io import BytesIO
from PIL import Image


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()
    QWEN2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0]
            if 'mmtag' in self.version:
                init_msg = init_msg.replace("<image>", "").strip()
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            elif not init_msg.startswith("<image>"):
                init_msg = init_msg.replace("<image>", "").strip()
                messages[0] = (init_role, "<image>\n" + init_msg)
            else:
                messages[0] = (init_role, init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        elif self.sep_style == SeparatorStyle.QWEN2:
            start = '<|im_start|>'
            end = '<|im_end|>\n'
            ret = start + 'system\n' + self.system + end
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += start + role + "\n" + message + end
                else:
                    ret += start + role + "\n"
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def process_image(self, image, image_process_mode, return_pil=False, image_format='PNG'):
        if image_process_mode == "Pad":
            def expand2square(pil_img, background_color=(122, 116, 104)):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        max_len, min_len = 672, 448
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))
        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    if type(image) != list:
                        image = [image]
                    for img in image:
                        img = self.process_image(img, image_process_mode, return_pil=return_pil)
                        images.append(img)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    if type(image) != list:
                        image = [image]
                    if len(image) == 1:
                        msg = '<image>\n' + msg.replace('<image>', '').strip()
                    else:
                        msg = re.sub(r'(<image>)\n(?=<image>)', r'\1 ', msg)
                    for img in image:
                        img_b64_str = self.process_image(img, "Default", return_pil=False, image_format='JPEG')
                        img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}"/>'
                        msg = msg.replace('<image>', img_str, 1).strip()
                    if len(msg) > 0:
                        ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_qwen_v1 = Conversation(
    system="You are a helpful assistant.",
    roles=("user", "assistant"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.QWEN2,
)

conv_deepseek_v1 = Conversation(
    system="",
    roles=("User", "Assistant"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="\n",
    sep2="<｜end▁of▁sentence｜>",
)


conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mistral_instruct = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="",
    sep2="</s>",
)

conv_llava_llama_2_simple = Conversation(
    system="Answer the questions about the visual content that the user provides.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llava_llama_2_mmtag = Conversation(
    system="Answer the questions about the visual content that the user provides."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2_mmtag",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

conv_mistral_orca = Conversation(
    system="""<|im_start|>system
You are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_mistral_zephyr = Conversation(
    system="""<|system|>
You are a helpful AI assistant.""",
    roles=("<|user|>\n", "<|assistant|>\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="</s>",
)

conv_mistral_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

default_conversation = conv_vicuna_v1
conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "v1_deepseek": conv_deepseek_v1,
    'v1_qwen2': conv_qwen_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llama_2": conv_llama_2,
    "mistral_instruct": conv_mistral_instruct,
    "mistral_orca": conv_mistral_orca,
    "mistral_zephyr": conv_mistral_zephyr,
    "mistral_direct": conv_mistral_direct,

    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "llava_v0": conv_llava_v0,
    "llava_v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "llava_v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,
    "llava_llama_2_simple": conv_llava_llama_2_simple,
    "llava_llama_2_mmtag": conv_llava_llama_2_mmtag,
    "llava_mistral_instruct": conv_mistral_instruct,

    "mpt": conv_mpt,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())


    convs = [
            {
                "from": "human",
                "value": "写一段图文并茂的文章，文章的开头是：\n向人民汇报，创人民满意！近日气温骤降，雨雪天气频现，不仅给出行带来不便，更存在很多安全隐患。尤其是大雾、团雾的出现，更是给道路交通带来危险。不过，不管天气多么恶劣，风里雨里安徽公安都会陪伴着您，守护人民的平安出行……中央气象台12月22日06时发布大雾橙色预警：预计22日早晨至白天，河北南部、山东西部和南部、河南北部和东部、安徽北部、江苏北部、湖北中部等地有大雾，其中，河北东南部、山东西部、河南北部等地的部分地区有能见度不足200米的强浓雾，局地有能见度不足50米的特强浓雾。@江淮气象今晨消息，安徽省发布大雾黄色预警，12小时内江北部分地区将出现能见度小于1000米的雾，局部地区小于200米。请注意防范！雨雾天气 务必注意出行安全雨雾天气路面湿滑能见度低给道路交通带来安全隐患<image>\n"
            },
            {
                "from": "gpt",
                "value": "尤其是团雾出现<image>\n更是高速通行的“流动杀手”随着天气转冷，昼夜温差加大，团雾天气增多，特别是在夜间和清晨尤其多发。一旦在高速上遇到团雾，应该采取哪些措施来避免事故呢？不要慌，保持警惕在高速上突然飘来团雾，视野甚至瞬间消失的时候，千万不能慌张的急刹停车，这种操作是最危险的，因为高速车速极快，后车极大可能会追尾上来，这时候要稳住方向盘，采用点刹的方式将车速降下来。<image>\n及时打开雾灯在降低车速的同时，还要及时打开双闪和雾灯，提醒周围车辆。雾灯的光源强度能穿透浓雾，让后车在较远的距离就能确定你车辆的位置，及时采取制动措施。要注意的是，不要随便打开远光灯，因为其光线照射到空气中的浮尘会扩散，车前白茫茫一片，严重影响驾驶视野。加大车距遇到能见度低的雾霾，除了降低车速，还要加大安全车距，保证遇到突发情况有足够的反应时间和距离。当然，合理的车距应该视雾霾大小来调整，一般来说，能见度小于200米大于100米时，时速不得超过60公里，保持100米以上的车距；能见度小于100米大于50米时，时速不得超过40公里，保持50米以上的车距；能见度在50米以内时，时速就要控制在20公里以下。做好以上三点以后，我们在确保安全的情况下，缓慢驶入右侧车道，尽快找到收费站或者服务区等出口驶离高速主线。当然，为了保险起见，冬季开车尽量避免夜里或早上走高速，因为天冷的时候这两个时间段很容易形成团雾，太阳出来的时候，团雾也将逐渐消失。<image>\n截至目前，在全省高速公路团雾多发路段己建成24<image>\n套雾区诱导防撞系统，"
            }
        ]

    conv = conv_templates['v1_qwen2']
    conv.messages = []
    conv.append_message('human', convs[0]['value'])
    conv.append_message('gpt', convs[1]['value'])

    print(conv.get_prompt())

