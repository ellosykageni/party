import torch
from PIL import Image
import gradio as gr
import chromadb
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity
import time

client = chromadb.Client()
collection = client.create_collection('prac')

image_paths = [
    'images/acient.jpg',
    'images/allan.jpg',
    'images/bar.jpg',
    'images/bar2.jpg',
    'images/beach.jpg',
    'images/black bird.jpg',
    'images/building.jpg',
    'images/car.jpg',
    'images/church.jpg',
    'images/coffee.jpg',
    'images/cycle.jpg',
    'images/drone.jpg',
    'images/fort.jpg',
    'images/grad.jpg',
    'images/ivingroom.jpg',
    'images/jaz.jpg',
    'images/jeo.jpg',
    'images/laptop.jpg',
    'images/leaves.jpg',
    'images/lion.jpg',
    'images/matrix.jpg',
    'images/moon.jpg',
    'images/muslim.jpg',
    'images/new.jpg',
    'images/raster man.jpg',
    'images/ree.jpg',
    'images/road.jpg',
    'images/rollercoster.jpg',
    'images/samsung memory.jpg',
    'images/snow.jpg',
    'images/sportcar.jpg',
    'images/sunset.jpg',
    'images/wedding.jpg',
]

model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

images = [Image.open(image_path) for image_path in image_paths]
inputs = processor(images=images, return_tensors ='pt', padding=True)
#print(inputs)

start_ingestion_time = time.time()

with torch.no_grad():
    image_embeddings = model.get_image_features(**inputs).numpy()

image_embeddings = [embeddings.tolist() for embeddings in image_embeddings]

end_ingestion_time = time.time()
ingestion_time = end_ingestion_time - start_ingestion_time

collection.add(
    embeddings = image_embeddings,
    metadatas = [{"image":image_path} for image_path in image_paths],
    ids = [str(i) for i in range(len(image_paths))]
)

def calculate_accuracy(image_embedding, query_embedding):
    similarity = cosine_similarity([image_embedding], [query_embedding])[0][0]
    return similarity

def search_query(query):
    if not query.strip():
        return None, "Hey 🙋‍♀️, did you forget something?" ""

        print(f"Query: {query}")

    query_time = time.time()
    qinputs = processor(text=query, return_tensors="pt", padding = True)
    
    with torch.no_grad():
        query_embeddings = model.get_text_features(**qinputs).numpy()
    
    query_embeddings = query_embeddings.tolist()


    results = collection.query(
        query_embeddings = query_embeddings,
        n_results = 1
    )

    result_image_path = results['metadatas'][0][0]['image']
    matched_image_index = int(results['ids'][0][0])
    matched_image_embedding = image_embeddings[matched_image_index]

    accuracy_score = calculate_accuracy(matched_image_embedding, query_embeddings[0])

    end_query_time = time.time()
    query_ingestion_time = end_query_time - query_time

    result_image = Image.open(result_image_path)

    file_name = result_image_path.split("/")[-1]
    return result_image, f"accuracy_score : {accuracy_score:.4f} query time {query_ingestion_time:.4f}seconds", file_name

squeries = [
        "A beautiul lady",
        'A vintage car',
        'interior design',
        'Joy Patience',
        'A graduate',
        'roads'
    ]

def populate_query(suggested_query):
        return suggested_query

with gr.Blocks() as gr_interface:
    gr.Markdown("# A VECTOR SEARCH PROJECT BY ALLAN MUNENE 🎉🎊")
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"**Image ingestion time**: {ingestion_time:.4f} seconds")

            gr.Markdown('### Input panel')
            custom_query = gr.Textbox(placeholder = "input your query here", label ="What are you looking for?")

            with gr.Row():
                submit_button = gr.Button("Submit Query")
                cancel_button = gr.Button("cancel")

            
            gr.Markdown('##### suggested search phrases')
            with gr.Row(elem_id = 'button-container'):
                for query in squeries:
                    gr.Button(query).click(fn=lambda q=query :q, outputs=custom_query)

        with gr.Column():
            gr.Markdown('### Retrived Images')
            image_result = gr.Image(type = 'pil', label = 'Result image')
            accuracy_measure= gr.Textbox(label = 'Performance')

        submit_button.click(fn=search_query, inputs=custom_query, outputs=[image_result, accuracy_measure])
        cancel_button.click(fn=lambda: (None, ''), outputs = [image_result, accuracy_measure])

gr_interface.launch(share=True)


