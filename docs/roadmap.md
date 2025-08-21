# SteerLab Implementation Roadmap

This document outlines the phased development plan for the SteerLab project, based on the detailed technical design. The roadmap prioritizes a strong technical foundation, faithful research reproduction, and long-term extensibility.

---

### **Phase 1: Core Steering Functionality (The "Engine")**

**Objective:** To implement and rigorously validate the core activation steering mechanism.

* **Tasks:**
    1. **Implement CAA:** Write the Python script to implement the **Contrastive Activation Addition (CAA)** algorithm for vector computation.
    2. **Generate First Vector:** Use the CAA script to generate a steering vector for a single preference axis (e.g., 'cost') on a baseline model (e.g., Gemma 2B) and save it as a `.safetensors` file.
    3. **Build `SteerableModel` Wrapper:** Create the wrapper class that encapsulates a Hugging Face model and contains the logic for registering and, crucially, clearing forward hooks within a `try...finally` block.
    4. **Unit Testing:** Develop unit tests to verify that the hook mechanism correctly modifies activation values at the specified layers and with the correct scaling.
    5. **Qualitative Validation:** Create a simple script to generate text with and without steering to confirm the vector has the intended qualitative effect.

* **Deliverable:** A validated core library that can apply a pre-computed steering vector to a model's activations during inference.

---

### **Phase 2: Application Layer & SELECT Interface (The "Controls")**

**Objective:** To build the API server and the most straightforward user interaction mode, creating a functional end-to-end system.

* **Tasks:**
    1. **API Scaffolding:** Set up the FastAPI server, including the Pydantic schemas (`ChatRequest`, `ChatResponse`, etc.) and the API endpoints defined in the API reference.
    2. **Engine Integration:** Integrate the `SteerableModel` and a new `SteeringVectorManager` class into the API server's inference logic.
    3. **Implement SELECT Mode:** Develop the full backend logic for the SELECT mode, including the `/preferences/{user_id}` endpoints.
    4. **Build Demo UI:** Create a minimal Gradio or Streamlit UI to interact with the SELECT mode for testing and demonstration.

* **Deliverable:** A running FastAPI application that can serve steered responses based on user-defined preferences sent via a simple UI.

---

### **Phase 3: Advanced Interfaces & Computational Reproduction (The "Science")**

**Objective:** To implement the more complex, stateful user interfaces and to formally validate the implementation by reproducing the paper's computational experiments.

* **Tasks:**
    1. **Implement CALIBRATE Mode:** Develop the stateful backend logic for the calibration process, including session management and the iterative, binary-search-like algorithm for converging on a user's preference.
    2. **Implement LEARN Mode:** Implement the feedback-driven update logic for the LEARN mode. This must include developing the NLU component (e.g., a prompted LLM call) for analyzing user feedback to derive a sentiment/intent score.
    3. **Reproduce Experiments:** Write scripts to conduct the key computational experiments from the "Steerable Chatbots" paper (e.g., evaluating steering effectiveness across a range of strengths).
    4. **Validate Results:** Generate plots and metrics from the experiments and compare them against the published figures to validate the implementation's faithfulness.

* **Deliverable:** A fully-featured backend supporting all three interaction modes, with experimental results that validate the implementation against the original research.

---

### **Phase 4: Refinement for Extensibility & Documentation (The "Framework")**

**Objective:** To refactor the codebase into a clean, well-documented, and reusable framework that invites community contribution.

* **Tasks:**
    1. **Abstract Core Components:** Refactor key components (e.g., vector generation) behind abstract base classes to allow for future extension with new algorithms (like SAS or FGAA).
    2. **Generalize Model Handling:** Ensure the `SteerableModel` class is robust and can wrap any Hugging Face causal language model.
    3. **Document Extension Points:** Create clear documentation and tutorials explaining how a user can define a new preference axis, prepare data, and generate their own steering vectors.
    4. **Finalize Documentation:** Write a comprehensive `README.md`, add extensive code comments, and ensure all `docs/` files are accurate and complete.

* **Deliverable:** A polished, well-documented, and extensible framework ready for open-source release.
