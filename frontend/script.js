
const API_URL = "http://127.0.0.1:8000";

// Get HTML elements
const fraudForm = document.getElementById("fraudForm");

const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const resultContent = document.getElementById("resultContent");
const errorMessage = document.getElementById("errorMessage");
const analyzeBtn = document.getElementById("analyzeBtn");


// -----------------------------------------
// Form submission
// -----------------------------------------

fraudForm.addEventListener("submit", async function(event) {

    event.preventDefault();

    const requestData = {
        step: Number(document.getElementById("step").value),

        amount: Number(document.getElementById("amount").value),

        transaction_type:
            document.getElementById("transaction_type").value,

        origin_account_type:
            document.getElementById("origin_type").value,

        destination_account_type:
            document.getElementById("destination_type").value
    };

    // Basic frontend validation
    if (
        !Number.isFinite(requestData.amount) ||
        requestData.amount < 0 ||
        !Number.isInteger(requestData.step) ||
        requestData.step < 1 ||
        requestData.step > 743
    ) {
        showError("Please enter valid transaction details.");
        return;
    }

    await analyzeTransaction(requestData);
});


// -----------------------------------------
// Call FastAPI
// -----------------------------------------

async function analyzeTransaction(data) {

    showLoading();

    analyzeBtn.disabled = true;

    try {

        const response = await fetch(
            `${API_URL}/predict`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(data)
            }
        );

        const result = await response.json();

        if (!response.ok) {

            throw new Error(
                result.detail || "Prediction request failed."
            );
        }

        displayResult(result);

    } catch (error) {

        showError(
            error.message === "Failed to fetch"
                ? "Cannot connect to the backend. Check whether FastAPI is running."
                : error.message
        );

        console.error("Prediction error:", error);

    } finally {

        analyzeBtn.disabled = false;
    }
}


// -----------------------------------------
// Display prediction
// -----------------------------------------

function displayResult(result) {

    hideAllStates();

    resultContent.classList.remove("hidden");

    const isFraud =
        result.prediction === "Fraudulent Transaction";

    const score = Number(result.fraud_score);
    const threshold = Number(result.threshold);

    // Prediction banner
    const banner = document.getElementById("resultBanner");

    banner.className =
        isFraud
            ? "result-banner fraud"
            : "result-banner legitimate";

    document.getElementById("predictionText").textContent =
        result.prediction;

    document.getElementById("riskStatus").textContent =
        result.risk_status;

    // Fraud score
    document.getElementById("fraudScore").textContent =
        `${score.toFixed(2)}%`;

    document.getElementById("thresholdValue").textContent =
        `${(threshold * 100).toFixed(0)}%`;

    const scoreBar = document.getElementById("scoreBar");

    scoreBar.style.width = `${Math.min(100, Math.max(0, score))}%`;

    scoreBar.className =
        isFraud ? "progress-fill fraud" : "progress-fill";

    // Transaction details
    document.getElementById("resultAmount").textContent =
        Number(result.transaction_details.amount).toLocaleString(
            "en-IN",
            {
                style: "currency",
                currency: "INR"
            }
        );

    document.getElementById("resultType").textContent =
        result.transaction_details.type;

    document.getElementById("resultStep").textContent =
        result.transaction_details.step;
}


// -----------------------------------------
// UI state functions
// -----------------------------------------

function hideAllStates() {

    emptyState.classList.add("hidden");
    loadingState.classList.add("hidden");
    resultContent.classList.add("hidden");
    errorMessage.classList.add("hidden");
}


function showLoading() {

    hideAllStates();

    loadingState.classList.remove("hidden");
}


function showError(message) {

    hideAllStates();

    errorMessage.textContent = message;

    errorMessage.classList.remove("hidden");
}


// -----------------------------------------
// Initial API connection check
// -----------------------------------------

async function checkAPI() {

    const statusElement =
        document.querySelector(".system-status");

    try {

        const response = await fetch(`${API_URL}/health`);

        if (!response.ok) {
            throw new Error("API unavailable");
        }

        const health = await response.json();

        if (!health.model_loaded) {
            throw new Error("Model unavailable");
        }

        statusElement.innerHTML =
            '<span class="status-dot"></span> API Connected';

    } catch (error) {

        statusElement.textContent = "API Disconnected";

        statusElement.style.color = "#c03939";
    }
}

checkAPI();
