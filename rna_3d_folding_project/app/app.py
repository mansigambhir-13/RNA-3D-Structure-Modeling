import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from api import RNA3DFoldingAPI

def convert_to_pdb(sequence, coords):
    """
    Convert RNA sequence and coordinates to PDB format.
    
    Args:
        sequence (str): RNA sequence (e.g., "GGAUCCGAUCC").
        coords (np.ndarray): 3D coordinates of the RNA structure (shape: [n, 3]).
    
    Returns:
        str: PDB-formatted string.
    """
    pdb_lines = []
    for i, (base, coord) in enumerate(zip(sequence, coords), start=1):
        pdb_lines.append(
            f"ATOM  {i:5d}  {base:3s} RNA A{i:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           {base[0]}"
        )
    return "\n".join(pdb_lines)

st.set_page_config(layout="wide", page_title="RNA 3D Structure Explorer")

st.title("RNA 3D Structure Explorer")

# Initialize API

@st.cache_resource
def load_model():
    try:
        return RNA3DFoldingAPI()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Two columns layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input")
    input_type = st.radio("Input type:", ["Single Sequence", "Multiple Sequences", "Example Dataset"])
    
    if input_type == "Single Sequence":
        sequence = st.text_area("Enter RNA sequence:", "GGAUCCGAUCC", height=100)
        color_by = st.selectbox("Color by:", ["Sequence", "Position", "Base Type"])
        
        if st.button("Predict Structure"):
            with st.spinner("Predicting structure..."):
                result = api.predict_structure(sequence)
                
                # Store in session state for access in the other column
                st.session_state.coords = np.array(result['coordinates'])
                st.session_state.sequence = sequence
                st.session_state.color_by = color_by
    
    elif input_type == "Multiple Sequences":
        uploaded_file = st.file_uploader("Upload CSV file with sequences", type=["csv"])
        if uploaded_file and st.button("Predict Structures"):
            with st.spinner("Processing sequences..."):
                df = pd.read_csv(uploaded_file)
                results = api.batch_predict(df)
                st.dataframe(results)
                
                # Allow downloading results
                csv = results.to_csv(index=False)
                st.download_button("Download Results", csv, "rna_predictions.csv")
    
    elif input_type == "Example Dataset":
        example = st.selectbox("Select example:", ["tRNA", "Ribosomal RNA fragment", "miRNA precursor"])
        examples = {
            "tRNA": "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCA",
            "Ribosomal RNA fragment": "GGGCUACGUCCUCGCGCCGCGCGUAACAACCCCAG",
            "miRNA precursor": "UGAGGUAGUAGGUUGUAUAGUU"
        }
        sequence = examples[example]
        if st.button("Load Example"):
            with st.spinner("Predicting structure..."):
                result = api.predict_structure(sequence)
                st.session_state.coords = np.array(result['coordinates'])
                st.session_state.sequence = sequence
                st.session_state.color_by = "Base Type"

with col2:
    st.subheader("3D Structure Visualization")
    
    if 'coords' in st.session_state:
        coords = st.session_state.coords
        sequence = st.session_state.sequence
        color_by = st.session_state.color_by
        
        # Create color mapping
        if color_by == "Sequence":
            colors = ['rgb(255,0,0)'] * len(sequence)
        elif color_by == "Position":
            colors = [f'rgb({int(255*i/len(sequence))},{int(255*(1-i/len(sequence)))},0)' for i in range(len(sequence))]
        elif color_by == "Base Type":
            color_map = {'A': 'rgb(255,0,0)', 'G': 'rgb(0,255,0)', 'C': 'rgb(0,0,255)', 'U': 'rgb(255,255,0)'}
            colors = [color_map[base] for base in sequence]
        
        # Create 3D plot
        fig = go.Figure(data=[
            # Line connecting residues
            go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                line=dict(color='grey', width=2),
                marker=dict(size=0),
                showlegend=False
            ),
            # Nucleotides as spheres
            go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                mode='markers+text',
                marker=dict(
                    size=8,
                    color=colors,
                ),
                text=[f"{i+1}:{base}" for i, base in enumerate(sequence)],
                hoverinfo='text'
            )
        ])
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download options
        st.subheader("Download Options")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Download as PDB"):
                # Function to convert to PDB format
                pdb_text = convert_to_pdb(sequence, coords)
                st.download_button("Download PDB", pdb_text, f"rna_structure.pdb")
        with col2:
            if st.button("Download as CSV"):
                df = pd.DataFrame({
                    'residue': list(sequence),
                    'position': range(1, len(sequence)+1),
                    'x': coords[:, 0],
                    'y': coords[:, 1],
                    'z': coords[:, 2]
                })
                st.download_button("Download CSV", df.to_csv(index=False), "rna_coordinates.csv")
        with col3:
            if st.button("Download Visualization"):
                # Create high-res figure for download
                st.markdown("Right-click on the plot and select 'Download as PNG'")