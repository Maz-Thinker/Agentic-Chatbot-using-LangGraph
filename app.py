from backend import (
    chatbot,
    generate_title,
    save_thread,
    get_all_threads,
    ingest_rag_document
)

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

import streamlit as st
import uuid
import tempfile
import os


DEFAULT_TITLE = "Nouvelle conversation"


# ============================================================
# Utility functions
# ============================================================

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id, title=DEFAULT_TITLE):

    if thread_id not in st.session_state["chat_threads"]:

        st.session_state["chat_threads"][thread_id] = title


def reset_chat():

    st.session_state["thread_id"] = generate_thread_id()

    st.session_state["message_history"] = []

    add_thread(
        st.session_state["thread_id"]
    )


def make_config(thread_id):

    return {
        "configurable": {
            "thread_id": thread_id
        },

        "metadata": {
            "thread_id": thread_id
        },

        "run_name": st.session_state[
            "chat_threads"
        ][thread_id]
    }


def load_conversation(thread_id):

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get(
        "messages",
        []
    )


# ============================================================
# HITL
# ============================================================

def get_pending_interrupt(config):

    """
    Check if the LangGraph execution is currently paused
    because of an interrupt().
    """

    state = chatbot.get_state(
        config=config
    )

    # LangGraph stores pending interrupts inside tasks
    for task in state.tasks:

        if task.interrupts:

            return task.interrupts[0].value

    return None


def display_interrupt_message(interrupt_value):

    """
    Display the information sent by interrupt().
    """

    st.warning(
        "⚠️ Human approval required"
    )

    if isinstance(
        interrupt_value,
        str
    ):

        st.markdown(
            f"### 🔐 Approval request\n\n"
            f"**{interrupt_value}**"
        )

    elif isinstance(
        interrupt_value,
        dict
    ):

        st.markdown(
            "### 🔐 Approval request"
        )

        message = interrupt_value.get(
            "message"
        )

        if message:

            st.markdown(
                f"**{message}**"
            )

        else:

            for key, value in interrupt_value.items():

                st.markdown(
                    f"**{key}:** {value}"
                )


# ============================================================
# Stream LangGraph
# ============================================================

def stream_graph(
    input_data,
    config
):

    """
    Run the LangGraph and stream AI messages.

    Returns:
        full_response
    """

    full_response = ""

    called_tools = set()

    with st.chat_message("assistant"):

        status_placeholder = st.empty()

        response_placeholder = st.empty()

        for message_chunk, metadata in chatbot.stream(

            input_data,

            config=config,

            stream_mode="messages"

        ):

            # ==================================================
            # AI message
            # ==================================================

            if isinstance(
                message_chunk,
                AIMessage
            ):

                # ==============================================
                # Detect tool call
                # ==============================================

                if message_chunk.tool_call_chunks:

                    for tc in message_chunk.tool_call_chunks:

                        name = tc.get(
                            "name"
                        )

                        if (
                            name
                            and name not in called_tools
                        ):

                            called_tools.add(
                                name
                            )

                            status_placeholder.markdown(
                                f"🔧 *Utilisation de `{name}`...*"
                            )

                # ==============================================
                # Display generated text
                # ==============================================

                if message_chunk.content:

                    status_placeholder.empty()

                    full_response += (
                        message_chunk.content
                    )

                    response_placeholder.markdown(
                        full_response
                    )

    return full_response


# ============================================================
# Initialization
# ============================================================

st.title(
    "Agentic Chatbot with LangGraph"
)


if "message_history" not in st.session_state:

    st.session_state[
        "message_history"
    ] = []


if "thread_id" not in st.session_state:

    st.session_state[
        "thread_id"
    ] = generate_thread_id()


if "chat_threads" not in st.session_state:

    st.session_state[
        "chat_threads"
    ] = dict(
        reversed(
            get_all_threads()
        )
    )


add_thread(
    st.session_state[
        "thread_id"
    ]
)


# ============================================================
# Current configuration
# ============================================================

tid = st.session_state[
    "thread_id"
]

CONFIG = make_config(
    tid
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title(
    "My Conversations"
)


# ============================================================
# New Chat
# ============================================================

if st.sidebar.button(
    "New Chat"
):

    reset_chat()

    st.rerun()


# ============================================================
# Conversations
# ============================================================

for thread_id, title in reversed(
    list(
        st.session_state[
            "chat_threads"
        ].items()
    )
):

    if st.sidebar.button(
        title,
        key=thread_id
    ):

        st.session_state[
            "thread_id"
        ] = thread_id

        messages = load_conversation(
            thread_id
        )

        temp_messages = []

        for message in messages:

            # ================================================
            # Human message
            # ================================================

            if isinstance(
                message,
                HumanMessage
            ):

                role = "user"

            # ================================================
            # AI message
            # ================================================

            elif isinstance(
                message,
                AIMessage
            ):

                role = "assistant"

            # ================================================
            # Ignore ToolMessage, etc.
            # ================================================

            else:

                continue

            temp_messages.append(
                {
                    "role": role,
                    "content": message.content
                }
            )

        st.session_state[
            "message_history"
        ] = temp_messages

        st.rerun()


# ============================================================
# Display chat history
# ============================================================

for message in st.session_state[
    "message_history"
]:

    with st.chat_message(
        message["role"]
    ):

        st.text(
            message["content"]
        )


# ============================================================
# CHECK FOR PENDING HITL INTERRUPT
# ============================================================

pending_interrupt = get_pending_interrupt(
    CONFIG
)


if pending_interrupt is not None:

    # ========================================================
    # Display approval request
    # ========================================================

    display_interrupt_message(
        pending_interrupt
    )

    st.markdown(
        "### What do you want to do?"
    )

    col1, col2 = st.columns(2)


    # ========================================================
    # APPROVE
    # ========================================================

    with col1:

        approve = st.button(
            "✅ Approve",
            use_container_width=True,
            key=f"approve_{tid}"
        )


    # ========================================================
    # REJECT
    # ========================================================

    with col2:

        reject = st.button(
            "❌ Reject",
            use_container_width=True,
            key=f"reject_{tid}"
        )


    # ========================================================
    # APPROVE ACTION
    # ========================================================

    if approve:

        with st.spinner(
            "Processing approved action..."
        ):

            response = stream_graph(

                Command(
                    resume="yes"
                ),

                CONFIG

            )


        # ====================================================
        # Save final assistant response
        # ====================================================

        if response:

            st.session_state[
                "message_history"
            ].append(
                {
                    "role": "assistant",
                    "content": response
                }
            )


        st.rerun()


    # ========================================================
    # REJECT ACTION
    # ========================================================

    if reject:

        with st.spinner(
            "Cancelling action..."
        ):

            response = stream_graph(

                Command(
                    resume="no"
                ),

                CONFIG

            )


        # ====================================================
        # Save final assistant response
        # ====================================================

        if response:

            st.session_state[
                "message_history"
            ].append(
                {
                    "role": "assistant",
                    "content": response
                }
            )


        st.rerun()


# ============================================================
# Chat input + File upload
# ============================================================

prompt = st.chat_input(
    "Type your message...",
    accept_file=True,
    file_type=["pdf"]
)


# ============================================================
# Process user input
# ============================================================

if prompt:

    user_input = prompt.text

    uploaded_files = prompt.files


    # ========================================================
    # Process uploaded files
    # ========================================================

    if uploaded_files:

        for uploaded_file in uploaded_files:

            temp_file_path = None

            try:

                # ============================================
                # Create temporary file
                # ============================================

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getvalue()
                    )

                    temp_file_path = (
                        temp_file.name
                    )


                # ============================================
                # Add PDF to FAISS knowledge base
                # ============================================

                ingest_rag_document(
                    temp_file_path
                )


                # ============================================
                # Delete temporary file
                # ============================================

                os.remove(
                    temp_file_path
                )


                # ============================================
                # Show uploaded file
                # ============================================

                with st.chat_message(
                    "user"
                ):

                    st.markdown(
                        f"📎 **{uploaded_file.name}**"
                    )


                # ============================================
                # Confirmation
                # ============================================

                with st.chat_message(
                    "assistant"
                ):

                    st.markdown(
                        f"✅ `{uploaded_file.name}` "
                        "has been added to the knowledge base."
                    )


            except Exception as e:

                # ============================================
                # Remove temporary file if necessary
                # ============================================

                if (
                    temp_file_path
                    and os.path.exists(
                        temp_file_path
                    )
                ):

                    os.remove(
                        temp_file_path
                    )


                st.error(
                    f"Error processing "
                    f"`{uploaded_file.name}`: {e}"
                )


    # ========================================================
    # Process text message
    # ========================================================

    if user_input:

        # ====================================================
        # Add user message to UI history
        # ====================================================

        st.session_state[
            "message_history"
        ].append(
            {
                "role": "user",
                "content": user_input
            }
        )


        # ====================================================
        # Display user message
        # ====================================================

        with st.chat_message(
            "user"
        ):

            st.text(
                user_input
            )


        # ====================================================
        # Refresh current thread/config
        # ====================================================

        tid = st.session_state[
            "thread_id"
        ]

        CONFIG = make_config(
            tid
        )


        # ====================================================
        # Run LangGraph
        # ====================================================

        response = stream_graph(

            {
                "messages": [
                    HumanMessage(
                        content=user_input
                    )
                ]
            },

            CONFIG

        )


        # ====================================================
        # Check if LangGraph is now interrupted
        # ====================================================

        pending_interrupt = get_pending_interrupt(
            CONFIG
        )


        if pending_interrupt is not None:

            # ==================================================
            # IMPORTANT:
            #
            # Do NOT save an empty assistant response.
            #
            # The graph is waiting for human approval.
            # ==================================================

            st.rerun()


        # ====================================================
        # Normal response
        # ====================================================

        if response:

            st.session_state[
                "message_history"
            ].append(
                {
                    "role": "assistant",
                    "content": response
                }
            )


        # ====================================================
        # Generate conversation title
        # ====================================================

        if (
            st.session_state[
                "chat_threads"
            ][tid]
            == DEFAULT_TITLE
            and response
        ):

            title = generate_title(
                user_input
            )


            st.session_state[
                "chat_threads"
            ][tid] = title


            save_thread(
                tid,
                title
            )


            st.rerun()