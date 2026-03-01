"""
Prompt Template for LongRefiner
================================

Handles prompt formatting and truncation for the LongRefiner algorithm.
"""


class PromptTemplate:
    """Prompt template handler for LongRefiner"""

    placeholders = ["reference", "question"]
    base_system_prompt = (
        "Answer the question based on the given document."
        "Only give me the answer and do not output any other words."
        "\nThe following are given documents.\n\n{reference}"
    )
    base_user_prompt = "Question: {question}"

    def __init__(self, tokenizer, system_prompt="", user_prompt=""):
        self.max_input_len = 64000
        self.tokenizer = tokenizer

        if len(system_prompt) == 0 and len(user_prompt) == 0:
            system_prompt = self.base_system_prompt
            user_prompt = self.base_user_prompt
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.enable_chat = True
        self.is_chat = True
        self.reference_template = None

    def truncate_prompt(self, prompt):
        """Truncate prompt to max_input_len"""
        assert isinstance(prompt, str)
        tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[
            0
        ]

        if len(tokenized_prompt) > self.max_input_len:
            print(
                f"The input text length is greater than the maximum length ({len(tokenized_prompt)} > {self.max_input_len}) and has been truncated!"
            )
            half = int(self.max_input_len / 2)
            prompt = self.tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + self.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        return prompt

    def get_prompt(
        self,
        question=None,
        retrieval_result=None,
        formatted_reference=None,
        previous_gen=None,
        messages=None,
        **params,
    ):
        """Generate prompt from inputs"""
        if messages is not None:
            if isinstance(messages, str):
                return self.truncate_prompt(messages)
            if self.is_chat and self.enable_chat:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                return self.truncate_prompt(prompt)
            prompt = "\n\n".join(
                [message["content"] for message in messages if message["content"]]
            )
            return self.truncate_prompt(prompt)

        if formatted_reference is None:
            if retrieval_result is not None:
                formatted_reference = self.format_reference(retrieval_result)
            else:
                formatted_reference = ""

        input_params = {"question": question, "reference": formatted_reference}
        input_params.update(**params)

        system_prompt = self.system_prompt.format(**input_params)
        user_prompt = self.user_prompt.format(**input_params)

        if self.is_chat and self.enable_chat:
            prompt_input = []
            if system_prompt != "":
                prompt_input.append({"role": "system", "content": system_prompt})
            if user_prompt != "":
                prompt_input.append({"role": "user", "content": user_prompt})
            prompt_input = self.tokenizer.apply_chat_template(
                prompt_input, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_input = "\n\n".join(
                [prompt for prompt in [system_prompt, user_prompt] if prompt != ""]
            )

        if previous_gen is not None and previous_gen not in ["", " "]:
            prompt_input += previous_gen

        return self.truncate_prompt(prompt_input)

    def format_reference(self, retrieval_result):
        """Format retrieval results as reference"""
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            if self.reference_template is not None:
                format_reference += self.reference_template.format(idx=idx, title=title, text=text)
            else:
                format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"

        return format_reference
