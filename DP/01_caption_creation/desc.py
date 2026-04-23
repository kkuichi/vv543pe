from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import gc
import pandas as pd
from tqdm import tqdm

#--------------------LOAD DATA------------------------
img_path = f""
captions_path = f""

#---------------CREATE CSV FOR RESULTS--------------------
if not os.path.exists(captions_path):
    pd.DataFrame(columns=["image_name", "caption"]).to_csv(captions_path, index=False)

print("model downloading")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

print("model's prepared for work")

#images = os.listdir(img_path)
#print(f"n_img:{len(images)}")

#------------------CHECK CREATED CAPTIONS------------------
df = pd.read_csv(captions_path)
done_img = set(df["image_name"].dropna().astype(str))
all_img = [f for f in os.listdir(img_path) if f.lower().endswith((".png")) ]
todo_img = [img for img in all_img if img not in done_img]

print(f"all img: {len(all_img)}")
print(f"done img: {len(done_img)}")
print(f"todo_img: {len(todo_img)}")


#----------------------PERFECT ANSWERS--------------------
#Príklady pre lepšiu kvalitu generovaných popisov
perfect_answer = [
    {
        "role": "system",
        "content": "You are an AI model that describes political memes visually and neutrally."
    },

    
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image_url": "https://www.themostlysimplelife.com/wp-content/uploads/2023/01/corn-trump.jpeg",
            },
            {
                "type": "text",
                "text": """Describe what is visually happening in the image in 3–5 sentences (around 70–120 words).
                           Focus only on what can be visually seen — do NOT mention or quote any text from the image.
                           Pay attention to general emotional tone, expressions, visual composition, relationships between elements."""
            },
        ],
    },
    {
        "role": "assistant",
        "content": """On the left, a person with light-colored hair styled upward and sideways appears to be caught in a moment of motion or wind, creating a slightly exaggerated, dynamic look. Their facial expression seems neutral to mildly serious, contrasting with the playful appearance of the hairstyle. On the right, there is an ear of corn with light green husks and fine, silky strands extending outward in a similar direction to the person’s hair. The juxtaposition emphasizes a humorous visual resemblance between the two subjects’ textures and forms."""
    },


    
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image_url": "https://miro.medium.com/v2/resize:fit:828/format:webp/0*7mFUsl4dHA7tvB2c.jpg",
            },
            {
                "type": "text",
                "text": """Describe what is visually happening in the image in 3–5 sentences (around 70–120 words).
                           Focus only on what can be visually seen — do NOT mention or quote any text from the image.
                           Pay attention to general emotional tone, expressions, visual composition, relationships between elements."""
            },
        ],
    },
    {
        "role": "assistant",
        "content": """The top part of the image shows a man in a suit leaning slightly forward, with a faint smile and a neutral or mildly friendly expression. The bottom part shows another man, also wearing a suit, with a confident or amused facial expression. The visual contrast between the two figures and their expressions creates a sense of comparison or sequence in tone and mood."""
    }
]

#----------------------GENERATE CAPTIONS------------------
for img_name in tqdm(todo_img, desc=f"process img", unit="img"):
    
        image_path = f"file://{os.path.join(img_path, img_name)}"

        model_answer = perfect_answer.copy()
        model_answer.append({
            "role": "user",
            "content": [
                {"type": "image", "image_url": image_path},
                {
                    "type": "text",
                    "text": """Describe what is visually happening in the image in 3–5 sentences (around 70–120 words).
                           Focus only on what can be visually seen — do NOT mention or quote any text from the image.
                           Pay attention to general emotional tone, expressions, visual composition, relationships between elements."""
                },
            ],
        })
    
        #transformácia vstupu do formátu vhodného pre model
        text = processor.apply_chat_template(model_answer, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(model_answer)
        #príprava tensorov pre model
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            tokens = model.generate(**inputs, max_new_tokens=128)

        #odstránenie vstupného promptu z výstupu
        tokens_without_prompt = [out_tokens[len(in_tokens):] for in_tokens, out_tokens in zip(inputs.input_ids, tokens)]
        #dekódovanie výstupu do textovej podoby
        caption = processor.batch_decode(tokens_without_prompt, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        print(f"\n {img_name}")
        print(f"caption: {caption[:100]}\n")

        #results.append({"image_name": img_name, "caption": caption})
        pd.DataFrame([{"image_name": img_name, "caption": caption}]).to_csv(captions_path, mode='a', header=False, index=False)

        del inputs, image_inputs, tokens, tokens_without_prompt
        torch.cuda.empty_cache()


print(f"\nbatch_{batch_id}[end], results:{captions_path}")
