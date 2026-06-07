const imageInput = document.getElementById("image");
const preview = document.getElementById("preview");
const placeholder = document.getElementById("placeholder");

if (imageInput) {
    imageInput.addEventListener("change", function () {
        const file = this.files[0];

        if (file) {
            const reader = new FileReader();

            reader.onload = function (event) {
                preview.src = event.target.result;
                preview.style.display = "block";
                placeholder.style.display = "none";
            };

            reader.readAsDataURL(file);
        }
    });
}

document.querySelectorAll(".bar-fill").forEach(function (bar) {
    const width = bar.getAttribute("data-width");

    if (width) {
        bar.style.width = width + "%";
    }
});