res <- function() {
    image <- magick::image_read("Face recognition/faces/test/user/user_46.png")
    faces <- image.libfacedetection::image_detect_faces(image)
    if (faces$nr == 1) {
        x <- faces$detections$x
        y <- faces$detections$y
        width <- faces$detections$width
        height <- faces$detections$height
    } else {
        print("Error: Can not recognise face!")
        break
    }
    faces <- magick::image_crop(image, geometry = magick::geometry_area(
        x = x,
        y = y,
        width = width,
        height = height
    ))
    faces <- as.integer(magick::image_data(faces))
    faces_tensor <- tensorflow::tf$convert_to_tensor(faces, dtype = tensorflow::tf$float32)
    faces_tensor <- tensorflow::tf$image$resize(faces_tensor, c(300L, 300L))
    faces_tensor <- tensorflow::tf$expand_dims(faces_tensor, axis = 0L)
    faces_tensor <- (faces_tensor / 127.5) - 1
    model <- keras3::load_model("Face recognition/faces/callbacks/best_7.keras")
    pred <- predict(model, faces_tensor)
    if (pred > 0.5) {
        print("User recognised")
    } else {
        print("Failed to recognise the user")
    }
}
