import os
import json

from dotenv import load_dotenv
from label_studio_sdk import Client
from label_studio_converter.brush import image2annotation

load_dotenv()

ls = Client(url=os.getenv("LABEL_STUDIO_URL"), api_key=os.getenv("API_KEY"))
ls.check_connection()

if (os.getenv("LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED") != 'true'):
    raise ValueError("Environment variable 'LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED' must be set to 'true' to import local files")

# create project specification
project = ls.start_project(
    title="Half-earth segmentation masks (512x512)",
    label_config="""
    <View>
        <Header value='$taxonomy' />
        <Image name='image' value='$image_url' />
        <BrushLabels name='tag' toName='image'>
            <Label value='specimen' background='#ff69b4' />
        </BrushLabels>
    </View>
    """,
)

project.connect_local_import_storage(os.path.join(os.getenv("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"), "output/"), 
                                                  regex_filter=".*png")

img_root = "output/resnet34-attention-ferns-512/transformed_img"
mask_root = "output/resnet34-attention-ferns-512/processed_masks"

img_files = os.listdir(img_root)

with open("output/halfearth-sample/sample-metadata.json", "r") as infile:
    metadata = {img['id']: {**img} for img in json.load(infile)}

tasks = [{
    "data": {
        "image_url": f"/data/local-files/?d={os.path.join(img_root, f)}",
        "taxonomy": f"{metadata[int(f.split('.')[0])]['species']} ({metadata[int(f.split('.')[0])]['family']})"
    },
    "annotations": [
        image2annotation(
            os.path.join(mask_root, f),
            label_name="specimen",
            from_name="tag",
            to_name="image"
    )],
} for f in img_files]

project.import_tasks(tasks)



