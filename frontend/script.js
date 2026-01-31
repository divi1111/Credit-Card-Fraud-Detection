// Update step slider value display
const stepSlider = document.getElementById('step');
const stepValue = document.getElementById('stepValue');

stepSlider.addEventListener('input', function() {
    stepValue.textContent = this.value;
});


// Transaction type mapping
const transactionTypes = {
    0: 'Cash In',
    1: 'Cash Out',
    2: 'Debit',
    3: 'Payment',
    4: 'Transfer'
};

const backendTransactionTypes = {
    0: "CASH_IN",
    1: "CASH_OUT",
    2: "DEBIT",
    3: "PAYMENT",
    4: "TRANSFER"
};

// Form submission handler
document.getElementById('fraudForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Get form values
    const senderName = document.getElementById('senderName').value.trim();
    const receiverName = document.getElementById('receiverName').value.trim();
    const step = parseInt(document.getElementById('step').value);
    const types = parseInt(document.getElementById('types').value);
    const amount = parseFloat(document.getElementById('amount').value);
    const oldbalanceorg = parseFloat(document.getElementById('oldbalanceorg').value);
    const newbalanceorg = parseFloat(document.getElementById('newbalanceorg').value);
    const oldbalancedest = parseFloat(document.getElementById('oldbalancedest').value);
    const newbalancedest = parseFloat(document.getElementById('newbalancedest').value);
    
    // Validate sender and receiver names
    if (!senderName || !receiverName) {
        showError('Error! Please input Transaction ID or Names of Sender and Receiver!');
        return;
    }
    
    // Calculate isflaggedfraud
    const isflaggedfraud = amount >= 200000 ? 1 : 0;
    
    // Prepare request data
    const requestData = {
    step: step,
    types: backendTransactionTypes[types], // 👈 STRING now
    amount: amount,
    oldbalanceorig: oldbalanceorg,
    newbalanceorig: newbalanceorg,
    oldbalancedest: oldbalancedest,
    newbalancedest: newbalancedest,
    isflaggedfraud: isflaggedfraud
};

    
    // Show loading spinner
    showLoading();
    
    try {
        // Make API request
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error('API request failed');
        }
        
        const result = await response.json();
        
        // Display transaction details
        displayTransactionDetails(
            senderName,
            receiverName,
            step,
            types,
            amount,
            oldbalanceorg,
            newbalanceorg,
            oldbalancedest,
            newbalancedest,
            isflaggedfraud
        );
        
        // Display prediction result
        displayPredictionResult(result.prediction, senderName, receiverName, types);
        
        hideLoading();
        
    } catch (error) {
        hideLoading();
        showError('Error connecting to the API. Please check if the API is available and try again.');
        console.error('Error:', error);
    }
});

function displayTransactionDetails(sender, receiver, step, type, amount, oldBalSender, newBalSender, oldBalRecipient, newBalRecipient, isFlagged) {
    const transactionType = transactionTypes[type];
    
    const detailsHTML = `
        <p><strong>Sender ID:</strong> ${sender}</p>
        <p><strong>Receiver ID:</strong> ${receiver}</p>
        <p><strong>1. Number of Hours it took to complete:</strong> ${step}</p>
        <p><strong>2. Type of Transaction:</strong> ${transactionType}</p>
        <p><strong>3. Amount Sent:</strong> $${amount.toLocaleString()}</p>
        <p><strong>4. Sender Balance Before Transaction:</strong> $${oldBalSender.toLocaleString()}</p>
        <p><strong>5. Sender Balance After Transaction:</strong> $${newBalSender.toLocaleString()}</p>
        <p><strong>6. Recipient Balance Before Transaction:</strong> $${oldBalRecipient.toLocaleString()}</p>
        <p><strong>7. Recipient Balance After Transaction:</strong> $${newBalRecipient.toLocaleString()}</p>
        <p><strong>8. System Flag Fraud Status (Transaction amount greater than $200000):</strong> ${isFlagged}</p>
    `;
    
    document.getElementById('transactionDetails').innerHTML = detailsHTML;
}

function displayPredictionResult(prediction, sender, receiver, type) {
    const transactionType = transactionTypes[type];
    const resultDiv = document.getElementById('predictionResult');
    
    resultDiv.textContent = `The '${transactionType}' transaction that took place between ${sender} and ${receiver} is ${prediction}.`;
    
    // Add appropriate styling based on prediction
    if (prediction.toLowerCase().includes('fraud')) {
        resultDiv.className = 'prediction-result fraud';
    } else {
        resultDiv.className = 'prediction-result legitimate';
    }
    
    // Show result container
    document.getElementById('resultContainer').classList.remove('hidden');
}

function showLoading() {
    document.getElementById('loadingSpinner').classList.remove('hidden');
    document.getElementById('resultContainer').classList.add('hidden');
    document.getElementById('errorMessage').classList.add('hidden');
}

function hideLoading() {
    document.getElementById('loadingSpinner').classList.add('hidden');
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
    document.getElementById('resultContainer').classList.add('hidden');
    document.getElementById('loadingSpinner').classList.add('hidden');
}