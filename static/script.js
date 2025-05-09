function openModal() {
    document.getElementById('accuracyModal').style.display = 'flex';
}

function closeModal() {
    document.getElementById('accuracyModal').style.display = 'none';
}

// Automatically show modal on result page load (optional)
window.onload = function() {
    openModal();
};
