#!/usr/bin/env python3
# gradio_ag.py: a python script that uses the gradio library to create a web
# app that allows users to predict age and gender based on given audio files.

from cProfile import label

import audiofile
import audresample
import gradio as gr

from predict_ag import predict_ag


def classify_speech(file, recording):
    """Function that takes in an audio file and returns the predicted age and
    gender of the speaker."""
    if file:
        audio, sr = audiofile.read(file)
        if sr != 16000:
            audio = audresample.resample(audio, sr, 16000)
        age, gender = predict_ag(audio, 16000)
        return age, gender


# audiofile_input = gr.inputs.Audio(type="filepath", label="Upload Audio File")
# demo = gr.Interface(
#     fn=classify_speech,
#     inputs=gr.Audio(type="filepath", label="Upload Audio File"),
#     # out 1 as age, output 2 as gender
#     outputs=["text", "label"],
#     examples=["./yachinene.wav"]
# )


if __name__ == "__main__":
    """Main function that launches the web app."""

    demo = gr.Blocks()
    with demo:
        gr.Markdown(
            """
            <center><h1>Age and Gender Prediction from Speech Using AI</h1></center>
            """
        )

        with gr.Row():
            # Inputs: choouse one, either upload file or use microphone
            with gr.Column():
                file_input = gr.Audio(type="filepath", label="Input")
                mic_input = gr.Mic(label="Input")
                
                with gr.Row():
                    clear_button = gr.Button(value="Clear", variant="secondary")
                    predict_button = gr.Button(value="Predict")


            # Outputs: Age (text) and Gender (Label and probability))
            with gr.Column():
                output_age = gr.Textbox(label="Age")
                output_gender = gr.Label(label="Gender")


        # Credits
        with gr.Row():
            gr.Markdown(
                """
                <center> <br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>
                <p>Created by Bagus Tris Atmaja <br/>
                Email: b-atmaja@aist.go.jp <br/>
                </p>
                </center?>
                """
            )


        # when clear button is clicked, clear the inputs
        clear_button.click(
            fn=gr.update,
            inputs=[],
            outputs=[file_input, mic_input, output_age, output_gender]
        )

        # when predict button is clicked, show the prediction
        predict_button.click(
            fn=classify_speech,
            inputs=[file_input, mic_input],
            outputs=[output_age, output_gender]
        )

        demo.launch(debug=True, share=True)
