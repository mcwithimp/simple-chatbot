import json

import boto3
import requests  # HTTP 요청 생성을 위한 requests 라이브러리 임포트
import streamlit as st  # Streamlit 라이브러리 임포트하여 웹 앱 생성

# Bedrock runtime
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# 웹 앱 제목 설정
st.title("Chatbot powered by Bedrock")

# 세션 상태에 메시지 없으면 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "inputTokenCount" not in st.session_state:
    st.session_state.inputTokenCount = 0

if "outputTokenCount" not in st.session_state:
    st.session_state.outputTokenCount = 0

# 세션 상태에 저장된 메시지 순회하며 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # 채팅 메시지 버블 생성
        st.markdown(message["content"])  # 메시지 내용 마크다운으로 렌더링


def chunk_handler(chunk):
    #  API가 서로 다른 타입을 리턴
    # print(f"\n\n!!!\n{chunk}")
    text = None
    chunk_type = chunk.get("type")
    if chunk_type == "content_block_start":
        # 응답 텍스트 시작
        text = chunk["content_block"]["text"]
    elif chunk_type == "content_block_delta":
        # 스트리밍 중인 응답 텍스트의 일부
        text = chunk["delta"]["text"]
    elif chunk_type == "message_stop":
        # 요청에 대한 메트릭을 포함
        metric = chunk["amazon-bedrock-invocationMetrics"]
        st.session_state.inputTokenCount = (
            st.session_state.inputTokenCount + metric["inputTokenCount"]
        )
        st.session_state.outputTokenCount = (
            st.session_state.outputTokenCount + metric["outputTokenCount"]
        )
        text = None
    else:
        text = None

    if text is not None:
        print(text, end="")
    return text


def get_streaming_response(prompt):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }
    jsonBody = json.dumps(body)

    # stream
    response = bedrock_runtime.invoke_model_with_response_stream(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=jsonBody,
    )
    stream = response.get("body")

    if stream:
        for event in stream:  # 스트림에서 반환된 각 이벤트 처리
            chunk = event.get("chunk")
            if chunk:
                chunk_json = json.loads(chunk.get("bytes").decode())
                text = chunk_handler(chunk_json)
                if text is not None:
                    yield text


# 사용자로부터 입력 받음
if prompt := st.chat_input("Message Bedrock..."):
    # 사용자 메시지 세션 상태에 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):  # 사용자 메시지 채팅 메시지 버블 생성
        st.markdown(prompt)  # 사용자 메시지 표시

    with st.chat_message("assistant"):  # 보조 메시지 채팅 메시지 버블 생성
        with st.spinner("AI 응답 생성중..."):
            model_output = st.write_stream(get_streaming_response(prompt))

    inputPrice = st.session_state.inputTokenCount / 1000 * 0.003
    outputPrice = st.session_state.outputTokenCount / 1000 * 0.015
    st.toast(f"{inputPrice + outputPrice} USD used", icon="💰")
    # 보조 응답 세션 상태에 추가
    st.session_state.messages.append({"role": "assistant", "content": model_output})
