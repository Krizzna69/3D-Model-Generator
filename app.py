import gradio as gr
from gradio_client import Client, handle_file
import time
from datetime import datetime

def log_message(message, status_box):
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    full_message = f"[{timestamp}] {message}"
    print(full_message)  # Console logging
    return full_message

def process_image_to_3d(input_image, progress=gr.Progress()):
    if input_image is None:
        return None, None, "Error: Please upload an image first"

    try:
        client = Client("JeffreyXiang/TRELLIS")
        
        # Step 1: Start session
        progress(0, desc="Initializing session...")
        client.predict(api_name="/start_session")
        
        # Step 2: Preprocess image
        progress(0.2, desc="Preprocessing image...")
        preprocessed = client.predict(
            image=handle_file(input_image),
            api_name="/preprocess_image"
        )
        yield None, None, "Image preprocessing complete..."
        
        # Step 3: Generate seed
        progress(0.3, desc="Generating seed...")
        seed = client.predict(
            randomize_seed=True,
            seed=0,
            api_name="/get_seed"
        )
        yield None, None, f"Seed generated: {seed}"
        
        # Step 4: Generate 3D model
        progress(0.4, desc="Generating 3D model...")
        result = client.predict(
            image=handle_file(preprocessed),
            multiimages=[],
            seed=seed,
            ss_guidance_strength=5.0,
            ss_sampling_steps=6,
            slat_guidance_strength=2,
            slat_sampling_steps=6,
            multiimage_algo="stochastic",
            api_name="/image_to_3d"
        )
        yield None, None, "3D model generation complete..."
        
        # Step 5: Extract GLB
        progress(0.8, desc="Creating GLB file...")
        glb_result = client.predict(
            mesh_simplify=0.98,
            texture_size=512,
            api_name="/extract_glb"
        )
        
        # Final result
        progress(1.0, desc="Processing complete!")
        yield (
            result["video"] if isinstance(result, dict) else result,
            glb_result[0] if isinstance(glb_result, tuple) else glb_result,
            "Processing complete! Your 3D model is ready."
        )
        
    except Exception as e:
        error_msg = str(e)
        if "GPU quota" in error_msg:
            yield None, None, """GPU quota exceeded. Please:
            1. Wait a few minutes and try again
            2. Create a free Hugging Face account
            3. Try again during off-peak hours"""
        else:
            yield None, None, f"Error occurred: {error_msg}"

# Create the Gradio interface
with gr.Blocks(title="3D Model Generator") as demo:
    current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    gr.Markdown("""
    # Image to 3D Model Converter
    
    ## Current Session Info:
    - User: Krizzna69
    - Time: {}
    
    ## Important Notes:
    - Please be patient during processing
    - Each step may take several minutes
    - You will see progress updates in the status box
    """.format(current_time))
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input Image",
                type="filepath"
            )
            status_output = gr.Textbox(
                label="Status",
                value="Ready to process...",
                max_lines=10
            )
            generate_btn = gr.Button("Generate 3D Model")
        
        with gr.Column():
            video_output = gr.Video(label="3D Preview")
            model_output = gr.Model3D(label="3D Model")

    def process_with_updates(image):
        current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{current_time}] Starting processing for user Krizzna69")
        
        for video, model, status in process_image_to_3d(image):
            print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] {status}")
            yield video, model, status

    generate_btn.click(
        fn=process_with_updates,
        inputs=[input_image],
        outputs=[video_output, model_output, status_output]
    )

if __name__ == "__main__":
    demo.queue()  # Enable queuing for real-time updates
    demo.launch(
        show_error=True,
        share=False,
        debug=True,
        server_port=7860,
        server_name="0.0.0.0"
    )