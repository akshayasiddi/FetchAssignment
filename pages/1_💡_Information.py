import streamlit as st
st.set_page_config(page_title="Information", page_icon="💡", layout="wide")
# Set the title for your app
st.title('Understanding Neural Network Parameters')

# Add some introductory text
st.write("""
         This page is designed to help you understand some of the terms used in 
         this app.
         """)

# Explain what an epoch is
st.header('Epoch')
st.write("""
         Think of an epoch like a single, complete run-through of all your data. If you're 
         learning to play a new song on the guitar, each epoch is like playing the song from 
         start to finish once. More epochs mean more practice, but too many might not make you 
         any better and just take up more of your time!
         """)


# Explain what the hidden layer size is
st.header('Hidden Layer Size')
st.write("""
         In our neural network, a hidden layer is like a team of detectives trying to solve a 
         mystery. Each detective (neuron) is good at figuring out one part of the puzzle. The 
         hidden layer size is how many detectives you have on the team. More detectives might 
         solve the puzzle faster, but too many might just get in each other's way.
         """)


# Explain what the sequence length is
st.header('Sequence Length')
st.write("""
         Let's say you're watching a TV series to guess what happens in the next episode. 
         The sequence length is like how many of the previous episodes you remember as you make 
         your guess. Remembering more episodes might help you guess better, but it might also 
         confuse you with too much information.
         """)
st.write("""
         In an ideal situation Epochs = 1000; Hidden Layer size = 5; Sequence Length = 3
         """)
\

# Add some closing text
st.write("""
         Play around with the settings to see how they affect the training!
         """)

