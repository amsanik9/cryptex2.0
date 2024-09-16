function predictAlgorithm() {
    const ciphertext = document.getElementById('ciphertext').value;
    if(ciphertext == '') 
        document.getElementById('result').textContent = 'PLEASE ENTER SOME TEXT!!'
    else {
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ ciphertext: ciphertext })
        })
        .then(response => response.json())
        .then(data => {
            showLoader();
            setTimeout(() => {
                hideLoader();
            }, 2000);
            document.getElementById('result').textContent = `The predicted cryptographic algorithm is: ${data.algorithm}`;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
}

function clearTextarea() {
    document.getElementById('ciphertext').value = '';
    document.getElementById('result').textContent = '';
}

function showLoader() {
    document.getElementById('loader').style.display = 'block';
    document.getElementById('result').style.display = 'none';
}


function hideLoader() {
    document.getElementById('loader').style.display = 'none';
    document.getElementById('result').style.display = 'block';
}