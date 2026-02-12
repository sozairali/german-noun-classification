function checkGender() {
    const nounInput = document.getElementById('noun-input').value;

    fetch('/predict_gender/' + nounInput, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        },
        //body: JSON.stringify({ noun: nounInput })
    })
    .then(response => response.json())
    .then(data => {
        const genderResult = document.getElementById('gender-result');
        genderResult.innerHTML = `${data.gender} ${data.noun} `;
    })
    .catch(error => console.error('Error:', error));
}