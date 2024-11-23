import streamlit as st
from app import get_pdf_text
from app import get_text_chunks
from app import get_vector_store
from app import user_input
from paper_reading import get_pdf_files
from paper_reading import get_txt_from_pdf
from paper_reading import model_2



def mof_synthesis_processing(df):
    """Placeholder for MOF synthesis processing - replace with actual implementation"""
    # This is a simplified placeholder - you'll need to implement the actual processing
    st.warning("Full MOF processing implementation needed")
    return model_2(df)

def main():
    st.set_page_config("Multi-Functional PDF Tool")
    
    # Sidebar for mode selection
    app_mode = st.sidebar.selectbox("Choose Application Mode", 
                                    ["Chat with PDF", "MOF Synthesis Paper Processing"])
    
    if app_mode == "Chat with PDF":
        st.header("Chat with PDF using Groq and HuggingFace Embeddings💁")
        
        user_question = st.text_input("Ask a Question from the PDF Files")

        if user_question:
            user_input(user_question)

        with st.sidebar:
            st.title("Upload PDFs:")
            pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDFs Processed Successfully")
    
    elif app_mode == "MOF Synthesis Paper Processing":
        st.title("MOF Synthesis Paper Processing")
        
        # Folder input for PDF processing
        folder = st.text_input("Enter the full path to the folder containing PDF files")
        
        if folder:
            try:
                folder_path = get_pdf_files(folder)
                
                if st.button("Process MOF Synthesis Papers"):
                    with st.spinner("Extracting text from PDFs..."):
                        # Extract text from PDFs
                        pdf_dataframe = get_txt_from_pdf(folder_path)
                    
                    if not pdf_dataframe.empty:
                        st.write(f"Extracted text from {len(pdf_dataframe['file name'].unique())} PDFs")
                        st.write(f"Total pages processed: {len(pdf_dataframe)}")
                        
                        # Process MOF synthesis papers
                        with st.spinner("Analyzing MOF Synthesis Papers..."):
                            processed_df = mof_synthesis_processing(pdf_dataframe)
                            
                            if not processed_df.empty:
                                # Display results
                                st.success("MOF Synthesis Papers Processed Successfully!")
                                
                                # Option to download processed data
                                csv = processed_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Processed Data",
                                    data=csv,
                                    file_name="mof_synthesis_data.csv",
                                    mime="text/csv"
                                )
                                
                                # Display first few rows
                                st.dataframe(processed_df)
                            else:
                                st.warning("No MOF synthesis data was extracted.")
                    
                    else:
                        st.error("No PDFs found or error in extracting text")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()