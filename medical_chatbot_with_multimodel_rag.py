from langchain.retrievers.multi_vector import MultiVectorRetriever

"""### Loading Connention to LLM"""

from dotenv import load_dotenv
load_dotenv()

import os
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

from langchain_openai import ChatOpenAI
chatgpt = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)

# Embedding Model
from langchain_openai import OpenAIEmbeddings

openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')

import uuid
from langchain_community.vectorstores import FAISS
from langchain_community.utilities.redis import get_client
from langchain_community.storage import RedisStore
vectorstore = FAISS.load_local("./faiss_index", 
                               embeddings=openai_embed_model,
                               allow_dangerous_deserialization=True)
# Initialize docstore (e.g., Redis)
client = get_client('redis://localhost:6379')
redis_store = RedisStore(client=client)
retriever = vectorstore.as_retriever()

import json

with open("text_chunks.json", "r", encoding="utf-8") as f:
    text_docs = json.load(f)

with open("captions.json", "r", encoding="utf-8") as f:
    image_captions = json.load(f)

with open("images_base64.json", "r", encoding="utf-8") as f:
    imgs_base64 = json.load(f)


"""## Multi-vector retriever"""
from langchain_core.documents import Document

def create_multi_vector_retriever(
    docstore, vectorstore,  texts, images
):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
    )

    # Helper to add to vectorstore & docstore
    def add_documents(retriever, doc_texts, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_texts)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    # Add text
    if texts:
        add_documents(retriever, text_docs, texts)
    # Add images
    if images:
        add_documents(retriever, image_captions, images)

    return retriever

# Build retriever
retriever_multi_vector = create_multi_vector_retriever(
    redis_store,
    vectorstore,
    text_docs,
    imgs_base64,
)

#query = "What is Achromatopsia?"
#docs = retriever_multi_vector.invoke(query, limit=5)


"""## To seperate retrieved content(images and texts)"""

import re
import base64

def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content.decode('utf-8')
        else:
            doc = doc.decode('utf-8')
        if looks_like_base64(doc) and is_image_data(doc):
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}

#query = "What kind of skin disorder in which the sebaceous glands become inflamed?"


"""
## Multimodal RAG
### Building End-to-End Multimodal RAG Pipeline
"""

from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

def multimodal_prompt_function(data_dict):
    """
    Create a multimodal prompt with both text and image context.

    This function formats the provided context from `data_dict`, which contains
    textand base64-encoded images. It joins the text portions
    and prepares the image(s) in a base64-encoded format to be included in a message.

    The formatted text and images (context) along with the user question are used to
    construct a prompt for GPT-4o
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    # Adding the text and tables for analysis
    text_message = {
        "type": "text",
        "text": (
            f"""You are an analyst tasked with understanding detailed information and trends
                from text documents and charts, stock photos and medical conditions in images.
                You will be given context information below which will be a mix of text
                and images usually of charts or medical conditions.
                Use this information to provide answers related to the user question.
                Analyze all the context information including  text and images to generate the answer.
                Do not make up answers, use the provided context documents below
                and answer the question to the best of your ability.

                User question:
                {data_dict['question']}

                Context documents:
                {formatted_texts}

                Answer:
            """
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


# Create RAG chain
multimodal_rag = (
        {
            "context": itemgetter('context'),
            "question": itemgetter('input'),
        }
            |
        RunnableLambda(multimodal_prompt_function)
            |
        chatgpt
            |
        StrOutputParser()
)

# Pass input query to retriever and get context document elements
retrieve_docs = (itemgetter('input')
                    |
                retriever_multi_vector
                    |
                RunnableLambda(split_image_text_types))


multimodal_rag_w_sources = (RunnablePassthrough.assign(context=retrieve_docs)
                                               .assign(answer=multimodal_rag)


)

#query = "Tell me detailed explanation for an achondroplastic."
#multimodal_rag_qa(query)
