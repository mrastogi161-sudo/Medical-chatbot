
# Load data page by page
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Medical_book.pdf")
pages = loader.load()


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

def save_text_and_image():
    # Extract and save images using Fitz
    import fitz
    import os

    output_folder = "extracted_images"
    os.makedirs(output_folder, exist_ok=True)

    doc = fitz.open('Medical_book.pdf')
    image_count = 0

    for page_number in range(len(doc)):
        page = doc[page_number]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"page{page_number+1}_img{img_index+1}.{image_ext}"
            with open(os.path.join(output_folder, image_filename), "wb") as f:
                f.write(image_bytes)
            image_count += 1

    print(f"Extracted and saved {image_count} images using PyMuPDF!")

    # Chunck texts
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

    documents = text_splitter.split_documents(pages)
    print(f"Total chunks: {len(documents)}")

    text_docs = [doc.page_content for doc in documents]

    ### Image Captioning

    # Using BLIP model to caption images and using them for retirver process
    from PIL import Image
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import os
    import base64

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def blip_caption(image_path):
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption

    def generate_img_summaries_blip(path):
        img_base64_list = []
        image_captions = []

        for img_file in sorted(os.listdir(path)):
            if img_file.endswith(".jpg"):
                img_path = os.path.join(path, img_file)
                # keep base64 for docstore
                with open(img_path, "rb") as img_file_obj:
                    base64_image = base64.b64encode(img_file_obj.read()).decode("utf-8")
                    img_base64_list.append(base64_image)

                caption = blip_caption(img_path)
                image_captions.append(caption)

        return img_base64_list, image_captions

    imgs_base64, image_captions = generate_img_summaries_blip('./extracted_images')

    """## Add raw data to vector store FAISS and database Redis"""

    import uuid
    from langchain_community.storage import RedisStore
    from langchain_community.utilities.redis import get_client
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings

    # Initialize empty FAISS index

    vectorstore = FAISS.from_documents(documents, embedding=OpenAIEmbeddings(api_key = OPENAI_KEY))
    
    # Saving images and texts chunks in json
    import json

    with open("text_chunks.json", "w", encoding="utf-8") as f:
        json.dump(text_docs, f)
        print("Chunking saved in json file.")

    with open("captions.json", "w", encoding="utf-8") as f:
        json.dump(image_captions, f)
        print("Image captions saved in json file.")

    with open("images_base64.json", "w", encoding="utf-8") as f:
        json.dump(imgs_base64, f)
        print("Raw images saved in json file.")

    vectorstore.save_local("./faiss_index")
    print("✅ FAISS index saved!")
    pass

if __name__ == "__main__":
    save_text_and_image()

