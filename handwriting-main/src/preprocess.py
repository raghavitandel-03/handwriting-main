def preprocess_image(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"\nImage not found:\n{image_path}"
        )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(image_path).convert("L")

    original = np.array(
        image,
        dtype=np.uint8
    )

    print()
    print("Original image size:")
    print(original.shape)

    # --------------------------------------------------------
    # Detect handwriting
    #
    # Dark pixels are treated as handwriting.
    # --------------------------------------------------------

    mask = original < 245

    coordinates = np.argwhere(mask)

    if coordinates.size == 0:
        raise ValueError(
            "No handwriting detected in image."
        )

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)

    # --------------------------------------------------------
    # Add margin around handwriting
    # --------------------------------------------------------

    margin = 5

    x_min = max(
        0,
        x_min - margin
    )

    y_min = max(
        0,
        y_min - margin
    )

    x_max = min(
        original.shape[1],
        x_max + margin + 1
    )

    y_max = min(
        original.shape[0],
        y_max + margin + 1
    )

    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------

    cropped = original[
        y_min:y_max,
        x_min:x_max
    ]

    print("Detected handwriting box:")
    print(
        f"x={x_min}:{x_max}, "
        f"y={y_min}:{y_max}"
    )

    print("Cropped image size:")
    print(cropped.shape)

    # --------------------------------------------------------
    # Preserve aspect ratio
    # --------------------------------------------------------

    cropped_image = Image.fromarray(
        cropped
    )

    crop_width, crop_height = (
        cropped_image.size
    )

    target_width = 246
    target_height = 54

    scale = min(
        target_width / crop_width,
        target_height / crop_height
    )

    new_width = max(
        1,
        int(crop_width * scale)
    )

    new_height = max(
        1,
        int(crop_height * scale)
    )

    cropped_image = cropped_image.resize(
        (
            new_width,
            new_height
        ),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # White 256 × 64 canvas
    # --------------------------------------------------------

    canvas = Image.new(
        "L",
        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT
        ),
        color=255
    )

    # --------------------------------------------------------
    # Center handwriting
    # --------------------------------------------------------

    paste_x = (
        IMAGE_WIDTH - new_width
    ) // 2

    paste_y = (
        IMAGE_HEIGHT - new_height
    ) // 2

    canvas.paste(
        cropped_image,
        (
            paste_x,
            paste_y
        )
    )

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    image = np.array(
        canvas,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image = image / 255.0

    # --------------------------------------------------------
    # Channel dimension
    # --------------------------------------------------------

    image = np.expand_dims(
        image,
        axis=-1
    )

    # --------------------------------------------------------
    # Batch dimension
    # --------------------------------------------------------

    image = np.expand_dims(
        image,
        axis=0
    )

    print("Final model input:")
    print(image.shape)

    return image