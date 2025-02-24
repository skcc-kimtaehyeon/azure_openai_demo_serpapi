import json
import re
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Optional

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from approaches.approach import Approach
import logging

class ChatApproach(Approach, ABC):

    NO_RESPONSE = "0"

    @abstractmethod
    async def run_until_final_call(self, messages, overrides, auth_claims, should_stream) -> tuple:
        pass

    def get_search_query(self, chat_completion: Any, user_query: str) -> str:
        """
        OpenAI 응답에서 검색어를 추출하고, 필요하면 SerpAPI를 사용하여 검색 수행.
        """
        if isinstance(chat_completion, dict):
            if chat_completion.get("object") == "chat.completion.chunk":
                if "choices" not in chat_completion or not chat_completion["choices"]:
                    logging.warning("⚠️ OpenAI 응답에 'choices'가 없음.")
                    return user_query
                response_delta = chat_completion["choices"][0].get("delta", {})
                # None 처리 추가
                query_text = str(response_delta.get("content", "")).strip()
                if query_text:
                    return query_text
                return user_query

            else:
                try:
                    chat_completion = ChatCompletion(**chat_completion)  # dict -> ChatCompletion 변환
                except Exception as e:
                    logging.error(f"ChatCompletion 변환 실패: {e}")
                    return user_query

        if not chat_completion.choices:
            logging.warning("OpenAI 응답에 'choices'가 누락.")
            return user_query
        response_message = chat_completion.choices[0].message
        if response_message.tool_calls:
            for tool in response_message.tool_calls:
                if tool.type != "function":
                    continue
                function = tool.function
                if function.name == "search_sources":
                    arg = json.loads(function.arguments)
                    search_query = arg.get("search_query", self.NO_RESPONSE)
                    if search_query != self.NO_RESPONSE:
                        return search_query
        elif query_text := response_message.content:
            if query_text.strip() != self.NO_RESPONSE:
                return query_text   
        return user_query



    def extract_followup_questions(self, content: Optional[str]):
        if content is None:
            return content, []
        return content.split("<<")[0], re.findall(r"<<([^>>]+)>>", content)

    async def run_without_streaming(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        session_state: Any = None,
    ) -> dict[str, Any]:
        
                
        user_query = messages[-1]["content"]

        # ✅ OpenAI가 검색 요청을 하면 SerpAPI를 실행하여 쿼리 수정
        
        
        # OpenAI API 호출        
        extra_info, chat_coroutine = await self.run_until_final_call(
            messages, overrides, auth_claims, should_stream=False
        )
        # OpenAI 응답 처리
        chat_completion_response: ChatCompletion = await chat_coroutine
        content = chat_completion_response.choices[0].message.content
        role = chat_completion_response.choices[0].message.role
        web_search_results = ""
        # web_search_results = self.search_with_serpapi(user_query)
        if overrides.get("useSerpAPI"): 
            logging.info(f"SerpAPI 검색어: {user_query}")
            web_search_results = self.search_with_serpapi(user_query)
            content += f"\n\n web 검색 결과 : {web_search_results}"
            extra_info["serpapi_search_results"] = web_search_results
            
        logging.info(f"검색 결과: {web_search_results}")
        # web_search_results = ""
        # if overrides.get("use_serpapi_search", True):
        #     search_query = self.get_search_query(chat_completion_response, user_query)
        #     if search_query != user_query:  # OpenAI가 새로운 검색어를 생성한 경우
        #         logging.info(f"OpenAI가 검색 요청을 감지함: {search_query}")
        #         web_search_results = self.search_with_serpapi(search_query)
        
        if overrides.get("suggest_followup_questions"):
            content, followup_questions = self.extract_followup_questions(content)
            extra_info["followup_questions"] = followup_questions
        chat_app_response = {
            "message": {"content": content, "role": role},
            "context": extra_info,
            "session_state": session_state,
        }
        return chat_app_response

    async def run_with_streaming(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        session_state: Any = None,
    ) -> AsyncGenerator[dict, None]:
        
        user_query = messages[-1]["content"]

        # OpenAI가 검색 요청을 하면 SerpAPI를 실행하여 쿼리 수정
        web_search_results = ""
        # web_search_results = self.search_with_serpapi(user_query)
        
        
        extra_info, chat_coroutine = await self.run_until_final_call(
            messages, overrides, auth_claims, should_stream=True
        )
        yield {"delta": {"role": "assistant"}, "context": extra_info, "session_state": session_state}

        followup_questions_started = False
        followup_content = ""
        chat_completion_response = None
        async for event_chunk in await chat_coroutine:
            # "2023-07-01-preview" API version has a bug where first response has empty choices
            event = event_chunk.model_dump()  # Convert pydantic model to dict
            if event["choices"]:
                completion = {
                    "delta": {
                        "content": event["choices"][0]["delta"].get("content"),
                        "role": event["choices"][0]["delta"]["role"],
                    }
                }
                # if event contains << and not >>, it is start of follow-up question, truncate
                content = completion["delta"].get("content")
                content = content or ""  # content may either not exist in delta, or explicitly be None
                
                if overrides.get("suggest_followup_questions") and "<<" in content:
                    followup_questions_started = True
                    earlier_content = content[: content.index("<<")]
                    if earlier_content:
                        completion["delta"]["content"] = earlier_content
                        yield completion
                    followup_content += content[content.index("<<") :]
                elif followup_questions_started:
                    followup_content += content
                else:
                    yield completion
                chat_completion_response = event
        if overrides.get("useSerpAPI"):
            logging.info(f"SerpAPI 검색어: {user_query}")
            web_search_results = self.search_with_serpapi(user_query)
            yield {"delta": {"role": "assistant", "content": f"\n\n**웹 검색 결과**\n{web_search_results}"}}
        logging.info(f"검색 결과: {web_search_results}")
        
        if followup_content:
            _, followup_questions = self.extract_followup_questions(followup_content)
            yield {"delta": {"role": "assistant"}, "context": {"followup_questions": followup_questions}}
        # if web_search_results:
        #     yield {"delta": {"role": "assistant"}, "context": {"serpapi_search_results": web_search_results}}

    async def run(
        self,
        messages: list[ChatCompletionMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> dict[str, Any]:
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})
        return await self.run_without_streaming(messages, overrides, auth_claims, session_state)

    async def run_stream(
        self,
        messages: list[ChatCompletionMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> AsyncGenerator[dict[str, Any], None]:
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})
        return self.run_with_streaming(messages, overrides, auth_claims, session_state)
