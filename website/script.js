document.getElementById("load-images-btn").addEventListener("click", function() {
    // Fetch the list of images
    fetch('/images')
        .then(response => response.json())
        .then(data => {
            const hiddenGallery = document.getElementById('hidden-gallery');

            data.images.forEach(filename => {
                const img = document.createElement('img');
                img.src = `/images/${filename}`;
                img.alt = filename;
                hiddenGallery.appendChild(img);

                // Further processing can be done here, e.g., loading into a canvas or passing to a model
                console.log(`Loaded image: ${filename}`);
            });

            alert('Images loaded for processing.');
        })
        .catch(error => console.error('Error loading images:', error));
});
