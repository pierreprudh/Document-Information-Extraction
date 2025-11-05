import streamlit as st
import asyncio
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import re
from io import BytesIO

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination

from dotenv import load_dotenv, find_dotenv
from PIL import Image
from pdf2image import convert_from_path
import sys

import tiktoken

ROOT = Path(__file__).resolve().parents[1]

try:
    from resources.prompts import Prompt, PRICING
except ModuleNotFoundError:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from resources.prompts import Prompt, PRICING

from datetime import datetime

found_env = find_dotenv(filename=".env", usecwd=True)
if found_env:
    load_dotenv(found_env)
else:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

class CostCalculator:
    """Class to calculate the costs of OpenAI requests"""

    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4o") -> int:
        """Count the number of tokens in a text"""
        try:
            # Use the appropriate encoding according to the model
            if "gpt-4" in model:
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in model:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")

            return len(encoding.encode(text))
        except Exception:
            # Rough estimate if tiktoken fails (1 token ≈ 4 characters)
            return len(text) // 4

    @staticmethod
    def estimate_cost(input_text: str, model: str, estimated_output_tokens: int = 500) -> Dict[str, float]:
        """Estimate the cost of a request"""
        # Clean the model name to match pricing
        model_key = model

        if model_key not in PRICING:
            model_key = "gpt-4o"  # Fallback

        pricing = PRICING[model_key]

        input_tokens = CostCalculator.count_tokens(input_text, model)

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (estimated_output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        return {
            "input_tokens": input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "model": model_key
        }

# Initialize cost tracking in session
if 'total_costs' not in st.session_state:
    st.session_state.total_costs = []
    st.session_state.cumulative_cost = 0.0

# Page configuration
st.set_page_config(
    page_title="Document data extraction",
    page_icon=":)",
    layout="wide"
)


def process_file_ocr(uploaded_file, ocr_engine: str) -> str:
    """Process the uploaded file with the selected OCR engine"""
    try:
        if uploaded_file.type == "application/pdf":
            # PDF processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            images = convert_from_path(tmp_file_path, dpi=150, thread_count=4)
            extracted_text = ""

            for i, img in enumerate(images, 1):
                st.write(f"🔎 Processing page {i}...")

                if ocr_engine == "Tesseract":
                    from resources.OCR import tesseract_ocr_image
                    text = tesseract_ocr_image(img)
                elif ocr_engine == "EasyOCR":
                    from resources.OCR import easyocr_process
                    text = easyocr_process(img)
                elif ocr_engine == "PaddleOCR":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                        img.save(tmp_img.name, format="PNG")
                        tmp_img_path = tmp_img.name
                    try:
                        from resources.OCR import paddleocr_process
                        text = paddleocr_process(tmp_img_path)
                    finally:
                        try:
                            os.remove(tmp_img_path)
                        except Exception:
                            pass
                elif ocr_engine == "Azure Document Intelligence":
                    from resources.OCR import azure_ocr
                    text = azure_ocr(img)

                extracted_text += f"\n--- Page {i} ---\n{text.strip()}\n"
            # Clean up temporary PDF file
            os.remove(tmp_file_path)
        else:
            # Image processing
            if ocr_engine == "Tesseract":
                from resources.OCR import tesseract_ocr_image
                extracted_text = tesseract_ocr_image(uploaded_file.getvalue())
            elif ocr_engine == "EasyOCR":
                from resources.OCR import easyocr_process
                extracted_text = easyocr_process(uploaded_file.getvalue())
            elif ocr_engine == "PaddleOCR":
                # Ensure we pass a filesystem path to the OCR helper
                data = uploaded_file.getvalue()
                try:
                    img = Image.open(BytesIO(data))
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                        img.save(tmp_img.name, format="PNG")
                        tmp_img_path = tmp_img.name
                except Exception:
                    # Fallback: write raw bytes directly
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix or ".img") as tmp_file:
                        tmp_file.write(data)
                        tmp_img_path = tmp_file.name
                try:
                    from resources.OCR import paddleocr_process
                    extracted_text = paddleocr_process(tmp_img_path)
                finally:
                    try:
                        os.remove(tmp_img_path)
                    except Exception:
                        pass
            elif ocr_engine == "Azure Document Intelligence":
                from resources.OCR import azure_ocr
                extracted_text = azure_ocr(uploaded_file.getvalue())

        if not extracted_text.strip():
            return "No text detected. Please check that the file contains readable text."

        cleaned_text = '\n'.join(line.strip() for line in extracted_text.splitlines() if line.strip())

        # --- Save OCR result to file in results directory ---
        results_dir = ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Nom d'origine + date derrière (sécurisé)
        original_name = Path(uploaded_file.name).stem
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', original_name)

        result_file = results_dir / f"{safe_name}_{timestamp}.txt"
        with result_file.open("w", encoding="utf-8") as f:
            f.write(cleaned_text)

        # Persist base path in session for later downloads/saves
        st.session_state["last_result_basepath"] = str(results_dir / f"{safe_name}_{timestamp}")

        return cleaned_text

    except Exception as e:
        return f"Error during OCR processing: {str(e)}"

def record_analysis_cost(ocr_text: str, model_name: str, processing_time: float = 0.0, system_prompt_text: Optional[str] = None):
    """Record the cost of an analysis in the session"""

    # Prepare the full prompt to calculate the actual cost
    system_message = system_prompt_text if system_prompt_text is not None else Prompt.get("Final", "")

    user_message = f"Here is the OCR extracted text:\n\n{ocr_text}\n\nPlease analyze it as instructed."
    full_input = f"{system_message}\n\n{user_message}"

    # Actual cost calculation
    cost_data = CostCalculator.estimate_cost(full_input, model_name)
    cost_data['timestamp'] = time.strftime("%H:%M:%S")
    cost_data['processing_time'] = processing_time

    # Record
    st.session_state.total_costs.append(cost_data)
    st.session_state.cumulative_cost += cost_data['total_cost']

    return cost_data

async def analyze_with_agent(ocr_text: str, model_name: str, api_key: str, system_prompt_text: str) -> Dict[str, Any]:
    """Analyze the OCR text with the selected agent"""
    try:
        client = OpenAIChatCompletionClient(
            model=model_name,
            api_key=api_key
        )

        extraction_agent = AssistantAgent(
            name="Extract_Expert",
            system_message=(system_prompt_text),
            model_client=client
        )

        termination = TextMentionTermination("TASK END")
        team = RoundRobinGroupChat([extraction_agent], termination_condition=termination)

        result = await team.run(
            task=f"Here is the OCR extracted text:\n\n{ocr_text}\n\nPlease analyze it as instructed."
        )

        return result

    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None

def main():
    st.title("Document Data Extractor")
    st.markdown("---")

    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")

        api_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", os.getenv("API_KEY_OPENAI", "")),
            type="password",
            help="Your OpenAI API key (env: OPENAI_API_KEY)",
            placeholder="sk-..."
        )

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        st.markdown("---")
        st.subheader("🔵 Azure Configuration (optionnal)")

        azure_endpoint = st.text_input(
            "Azure Document Intelligence Endpoint",
            value=os.getenv("AZURE_ENDPOINT", ""),
            help="URL de votre endpoint Azure Document Intelligence",
            placeholder="https://your-resource.cognitiveservices.azure.com/"
        )

        azure_key = st.text_input(
            "Azure Document Intelligence Key",
            value=os.getenv("AZURE_KEY", ""),
            type="password",
            help="API KEY Azure Document Intelligence"
        )
        if azure_endpoint:
            os.environ["AZURE_ENDPOINT"] = azure_endpoint
        if azure_key:
            os.environ["AZURE_KEY"] = azure_key


        st.markdown("---")

        # Model selection
        model_options = [
            "gpt-4.1-mini",
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "gpt-4.1"
        ]

        selected_model = st.selectbox(
            "🤖 Analysis model",
            model_options,
            index=0,
            help="Choose the OpenAI model for analysis"
        )

        # OCR selection
        ocr_options = ["Tesseract", "EasyOCR", "PaddleOCR", "Azure Document Intelligence"]
        selected_ocr = st.selectbox(
            "👁️ OCR engine",
            ocr_options,
            index=0,
            help="Choose the optical recognition engine"
        )

        st.markdown("---")
        st.markdown("### ℹ️ Information")
        st.markdown("**Supported formats:**")
        st.markdown("- Images : JPG, PNG, BMP, TIFF, GIF, WEBP")
        st.markdown("- Documents : PDF")

    # Main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📂 File upload")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif', 'webp', 'pdf'],
            help="Select an electricity bill to analyze"
        )

        if uploaded_file is not None:
            st.success(f"✅ File loaded: {uploaded_file.name}")
            st.session_state["last_uploaded_filename"] = uploaded_file.name

            # Preview display for images
            if uploaded_file.type != "application/pdf":
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="File preview", use_container_width=True)
                except:
                    st.warning("Unable to display preview")


        # Prompt selector (always displayed on the left column)
        prompt_options = list(Prompt.keys())
        selected_prompt_key = st.selectbox(
            "🧠 Analysis prompt",
            prompt_options,
            index=0,
            help="Choose the instruction used by the agent."
        )
        selected_prompt_text = Prompt[selected_prompt_key]

        # Optional: allow a custom prompt that overrides the selected preset
        use_custom_prompt = st.checkbox("✍️ Use a custom prompt")
        if use_custom_prompt:
            custom_prompt_text = st.text_area(
                "Custom prompt",
                value=st.session_state.get("custom_prompt_text", ""),
                height=260,
                help="This will override the selected preset above.")
            st.session_state.custom_prompt_text = custom_prompt_text
            if custom_prompt_text.strip():
                selected_prompt_text = custom_prompt_text

        # Preview of the selected prompt (just below the selector)
        st.markdown("**📄 Prompt preview**")
        st.text_area(
            label="Prompt preview",
            value=selected_prompt_text.strip(),
            height=260,
            label_visibility="collapsed",
            disabled=True,
            help="Text of the prompt currently used as system instruction."
        )

    with col2:
        st.header("📊 Analysis results")

        if uploaded_file is not None and api_key:
            if st.button("🚀 Start analysis", type="primary"):

                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Step 1: OCR
                    status_text.text("🔍 Extracting text...")
                    progress_bar.progress(25)

                    ocr_text = process_file_ocr(uploaded_file, selected_ocr)

                    if "Error" in ocr_text or "Erreur" in ocr_text:
                        st.error(ocr_text)
                        return



                    # Step 2: AI Analysis
                    status_text.text("🤖 AI analysis in progress...")
                    progress_bar.progress(50)

                    # Run async analysis
                    result = asyncio.run(analyze_with_agent(ocr_text, selected_model, api_key, selected_prompt_text))

                    # Record the analysis cost in a unified way
                    cost_data = record_analysis_cost(ocr_text, selected_model, processing_time=0.0, system_prompt_text=selected_prompt_text)

                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")

                    # Display results
                    if result and hasattr(result, 'messages') and result.messages:
                        st.markdown("### 📋 Extraction result")
                        # Display estimated cost just after the title with the unified logic
                        if cost_data:
                            st.markdown(f"**💸 Estimated request cost:** `${cost_data['total_cost']:.6f}` USD")
                        # Search for JSON in messages
                        json_result = None
                        for message in result.messages:
                            content = message.content.strip()

                            # Look for a valid JSON block in the text
                            match = re.search(r'\{.*\}', content, re.DOTALL)
                            if match:
                                try:
                                    json_result = json.loads(match.group(0))
                                    break
                                except json.JSONDecodeError:
                                    continue

                        if json_result:
                            # Generic, prompt-agnostic rendering of JSON results
                            st.markdown("### 📋 Extraction result")
                            # Estimated cost already shown above if available

                            # Split into scalar vs nested for a clean layout
                            flat_items = {k: v for k, v in json_result.items() if not isinstance(v, (dict, list))}
                            nested_items = {k: v for k, v in json_result.items() if isinstance(v, (dict, list))}

                            col_a, col_b = st.columns(2)

                            with col_a:
                                st.markdown("**🔎 Key fields**")
                                if flat_items:
                                    for k, v in flat_items.items():
                                        st.write(f"**{k}:** {v}")
                                else:
                                    st.write("No scalar fields detected.")

                            with col_b:
                                st.markdown("**📈 Highlights**")
                                # Show common metrics if present
                                if isinstance(json_result.get('total_amount', None), (int, float, str)):
                                    st.metric("Total amount", f"{json_result.get('total_amount')}")
                                if isinstance(json_result.get('sales_tax', None), (int, float, str)):
                                    st.metric("Sales tax / VAT", f"{json_result.get('sales_tax')}")
                                if isinstance(json_result.get('shipping_charges', None), (int, float, str)):
                                    st.metric("Shipping charges", f"{json_result.get('shipping_charges')}")
                                if isinstance(json_result.get('insurance', None), (int, float, str)):
                                    st.metric("Insurance", f"{json_result.get('insurance')}")

                                conf = json_result.get('confidence_score', None)
                                if isinstance(conf, (int, float)):
                                    st.metric("Confidence score", f"{int(conf)}%")
                                    st.progress(min(max(int(conf), 0), 100) / 100)

                            # Render nested structures (lists/dicts)
                            if nested_items:
                                st.markdown("### 🗂️ Structured sections")
                                for k, v in nested_items.items():
                                    st.markdown(f"**{k}**")
                                    try:
                                        # If it's a list of dicts, show as a table when possible
                                        if isinstance(v, list) and v and all(isinstance(it, dict) for it in v):
                                            st.dataframe(v)
                                        else:
                                            st.json(v)
                                    except Exception:
                                        st.json(v)

                            # Always provide access to the raw JSON
                            with st.expander("🔍 View full JSON"):
                                st.json(json_result)

                            st.session_state["last_json_result"] = json_result

                        else:
                            st.warning("No valid JSON result found")

                            # Display raw agent messages
                            with st.expander("📝 Agent messages"):
                                for i, message in enumerate(result.messages, 1):
                                    st.write(f"**Message {i} ({message.source}):**")
                                    st.code(message.content)
                    else:
                        st.error("No result obtained from analysis")

                    # Display OCR text in expander
                    with st.expander("📄 OCR extracted text"):
                        st.text_area("Content", ocr_text, height=200)

                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

        elif not api_key:
            st.warning("⚠️ Please configure your OpenAI API key")
        elif uploaded_file is None:
            st.info("📁 Please select a file to analyze")

        # --- Persistent Save JSON actions (no expander, direct buttons) ---
        if st.session_state.get("last_json_result"):
            json_result = st.session_state["last_json_result"]

            basepath = st.session_state.get("last_result_basepath")
            if basepath is None:
                results_dir = ROOT / "results"
                results_dir.mkdir(exist_ok=True)
                original_name = Path(st.session_state.get("last_uploaded_filename", "result")).stem
                safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', original_name)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                basepath = str(results_dir / f"{safe_name}_{timestamp}")

            suggested_name = f"{Path(basepath).name}_analysis.json"
            server_json_path = Path(basepath + "_analysis.json")

            col_left, col_right = st.columns(2)
            with col_left:
                st.download_button(
                    label="⬇️ Download extracted JSON",
                    data=json.dumps(json_result, ensure_ascii=False, indent=2),
                    file_name=suggested_name,
                    mime="application/json",
                    key="download_extracted_json_btn"
                )
            with col_right:
                if st.button("💾 Save extracted JSON to results folder", key="save_extracted_json_btn"):
                    (ROOT / "results").mkdir(exist_ok=True)
                    # Save the JSON result
                    with server_json_path.open("w", encoding="utf-8") as jf:
                        json.dump(json_result, jf, ensure_ascii=False, indent=2)
                    # Save the prompt used for analysis
                    prompt_path = Path(basepath + "_prompt.txt")
                    prompt_text = st.session_state.get("custom_prompt_text", None) if st.session_state.get("custom_prompt_text") else selected_prompt_text
                    # We want the prompt used for analysis (selected_prompt_text at time of analysis)
                    # If a custom prompt was used, it would have been set in selected_prompt_text already
                    # So we use selected_prompt_text here (from outer scope)
                    with prompt_path.open("w", encoding="utf-8") as pf:
                        pf.write(selected_prompt_text)
                    st.success(f"Saved: {server_json_path}\nand prompt: {prompt_path}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Developed by Pierreprudh | Powered by OpenAI, Tesseract, Azure </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
