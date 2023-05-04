import streamlit as st
from streamlit_chat import message
import boto3
from sagemaker.huggingface import HuggingFacePredictor

# Set up the SageMaker client
sagemaker = boto3.client('sagemaker')

# Call the list_endpoints API to get a list of endpoints
response = sagemaker.list_endpoints()

# Extract the list of endpoint names from the response
endpoint_names = [ep['EndpointName'] for ep in response['Endpoints']]

st.set_page_config(
    page_title="Streamlit Chat - Demo",
    page_icon=":robot:"
)

st.header("Chatbot Demo")

option = st.selectbox(
    'Select model endpoint',
    endpoint_names)

predictor = HuggingFacePredictor(endpoint_name=option)

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def query(payload):
    result = predictor.predict(payload)[0]
    response = f"{result['label'].title()} with {100*round(result['score'],2)}% confidence"
    return response

def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text

user_input = get_text()
 
if user_input:
    output = query(
        {
            "inputs": user_input
        })

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')