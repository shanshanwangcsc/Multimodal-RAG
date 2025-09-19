import os
from PIL import Image
from pdf2image import convert_from_path
from multiprocessing import Pool
import random

def process_pdf(args):
    pdf_path, output_folder = args
    try:
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        images = convert_from_path(pdf_path,dpi=300, thread_count=8, fmt='png')

        for image_id, image_data in enumerate(images):

            output_path = os.path.join(output_folder, f"{base_name}_page{image_id}.png")

            image_data.save(output_path, format="PNG")

    except Exception as e:
        print(f"Error processing '{pdf_path}': {e}")


def convert_pdfs_to_images_parallel(source_folder, output_folder):
    if not os.path.isdir(source_folder):
        raise FileNotFoundError(f"Source folder '{source_folder}' does not exist.")

    os.makedirs(output_folder, exist_ok=True)
    all_pdfs = [f for f in os.listdir(source_folder) if f.endswith(".pdf")]
    random.seed(42)

    pdf_files = random.sample(all_pdfs, min(10000, len(all_pdfs)))

    num_procs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print(f"Found {len(pdf_files)} PDF(s). Using up to {num_procs} CPU cores.")

    # Prepare args for each PDF

    args_list = [
    (os.path.join(source_folder, pdf_file), output_folder)
    for pdf_file in pdf_files
]



    with Pool(processes=num_procs) as pool:
        pool.map(process_pdf, args_list)


if __name__ == "__main__":
    source_folder = '/scratch/project_462000824/data/copernicus/pdf'
    output_folder = './images'
    convert_pdfs_to_images_parallel(source_folder, output_folder)
