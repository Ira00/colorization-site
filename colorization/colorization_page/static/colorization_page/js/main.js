var imageSelector = document.getElementById('image-selector');
var selectedImage = document.getElementById('selected-image');
var colorizeButton = document.getElementById('colorize');

imageSelector.addEventListener('change', function(event) {
  var file = event.target.files[0];
  var reader = new FileReader();

  reader.onload = function(event) {
    selectedImage.src = event.target.result;
  };

  reader.readAsDataURL(file);
  colorizeButton.classList.remove('d-none');
  colorizeButton.classList.add('d-all');
});