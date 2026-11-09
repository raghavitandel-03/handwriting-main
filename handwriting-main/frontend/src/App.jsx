import { useRef, useState } from "react";
import "./App.css";

function App() {
  const inputRef = useRef(null);

  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const selectImage = (file) => {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please choose an image file.");
      return;
    }

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setText("");
    setError("");
  };

  const handleFile = (e) => {
    selectImage(e.target.files[0]);
  };

  const recognize = async () => {
    if (!image) {
      setError("Please upload a handwriting image first.");
      return;
    }

    setLoading(true);
    setText("");
    setError("");

    try {
      const formData = new FormData();
      formData.append("image", image);

      const response = await fetch(
        "http://127.0.0.1:5000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Recognition failed");
      }

      setText(data.text || "");
    } catch (err) {
      console.error(err);

      if (
        err instanceof TypeError ||
        (err && err.message && err.message.includes("Failed to fetch"))
      ) {
        setError(
          "Backend server is not running. Start the Flask app on port 5000."
        );
      } else {
        setError("Unable to process the image.");
      }
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setImage(null);
    setPreview(null);
    setText("");
    setError("");

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  const copyText = async () => {
    if (!text) return;

    try {
      await navigator.clipboard.writeText(text);
    } catch {
      setError("Could not copy the text.");
    }
  };

  return (
    <div className="app">

      {/* HEADER */}
      <header className="header">

        <div className="brand">
          <div className="logo">
            ✎
          </div>

          <div>
            <h2>Handwriting Digitizer</h2>
            <p>Simple handwriting to text</p>
          </div>
        </div>

        <div className="header-link">
          Convert your handwriting
        </div>

      </header>


      {/* HERO */}
      <section className="hero">

        <div className="hero-badge">
          HANDWRITING DIGITIZER
        </div>

        <h1>
          Turn your handwriting
          <br />
          into <span>digital text.</span>
        </h1>

        <p>
          Upload a handwritten image and get clean,
          editable text in seconds.
        </p>

      </section>


      {/* MAIN WORKSPACE */}
      <main className="workspace">

        {/* LEFT */}
        <section className="card">

          <div className="card-header">

            <div>
              <div className="section-label">
                STEP 01
              </div>

              <h3>Upload handwriting</h3>

              <p>
                Choose an image of your handwriting.
              </p>
            </div>

          </div>


          {/* UPLOAD */}
          <div
            className={`upload-area ${
              preview ? "has-preview" : ""
            }`}
            onClick={() =>
              !preview && inputRef.current?.click()
            }
          >

            <input
              ref={inputRef}
              type="file"
              accept="image/png,image/jpeg,image/jpg"
              onChange={handleFile}
            />

            {!preview ? (
              <div className="upload-content">

                <div className="upload-icon">
                  <span>↑</span>
                </div>

                <h4>
                  Drop your image here
                </h4>

                <p>
                  or choose a file from your computer
                </p>

                <button
                  className="browse-button"
                  onClick={(e) => {
                    e.stopPropagation();
                    inputRef.current?.click();
                  }}
                >
                  Browse files
                </button>

                <span className="file-types">
                  JPG · JPEG · PNG
                </span>

              </div>
            ) : (

              <div className="preview-container">

                <img
                  src={preview}
                  alt="Uploaded handwriting"
                />

                <button
                  className="change-image"
                  onClick={(e) => {
                    e.stopPropagation();
                    inputRef.current?.click();
                  }}
                >
                  Change image
                </button>

              </div>

            )}

          </div>


          {/* FILE */}
          {image && (
            <div className="file-info">

              <div className="file-left">

                <div className="file-icon">
                  ◧
                </div>

                <div>
                  <strong>{image.name}</strong>

                  <span>
                    {(image.size / 1024).toFixed(1)} KB
                  </span>
                </div>

              </div>

              <button
                className="remove"
                onClick={clearAll}
              >
                Remove
              </button>

            </div>
          )}


          {/* ACTION */}
          <button
            className="convert-button"
            disabled={!image || loading}
            onClick={recognize}
          >

            {loading ? (
              <>
                <span className="spinner"></span>
                Processing...
              </>
            ) : (
              <>
                Convert to text
                <span>→</span>
              </>
            )}

          </button>


          {error && (
            <div className="error-message">
              <span>!</span>
              {error}
            </div>
          )}

        </section>


        {/* RIGHT */}
        <section className="card">

          <div className="card-header">

            <div>
              <div className="section-label">
                STEP 02
              </div>

              <h3>Your text</h3>

              <p>
                Edit the text if you need to make changes.
              </p>
            </div>

          </div>


          {/* RESULT */}
          <div className="result-area">

            {loading ? (

              <div className="loading-state">

                <div className="loading-circle"></div>

                <h4>
                  Converting handwriting
                </h4>

                <p>
                  This may take a moment.
                </p>

              </div>

            ) : text ? (

              <div className="result-content">

                <div className="result-toolbar">

                  <div className="completed">
                    <span>✓</span>
                    Completed
                  </div>

                  <button
                    className="copy-button"
                    onClick={copyText}
                  >
                    Copy text
                  </button>

                </div>


                <textarea
                  value={text}
                  onChange={(e) =>
                    setText(e.target.value)
                  }
                  spellCheck="false"
                />


                <div className="result-footer">

                  <span>
                    {text.length} characters
                  </span>

                  <button
                    onClick={() => setText("")}
                  >
                    Clear
                  </button>

                </div>

              </div>

            ) : (

              <div className="empty-state">

                <div className="empty-icon">
                  A
                </div>

                <h4>
                  Your text will appear here
                </h4>

                <p>
                  Upload an image and click
                  <br />
                  <strong>Convert to text</strong>.
                </p>

              </div>

            )}

          </div>

        </section>

      </main>


      {/* BOTTOM INFO */}
      <section className="bottom-info">

        <div className="info-item">
          <div className="info-icon">1</div>
          <div>
            <strong>Upload</strong>
            <p>Select your handwriting image.</p>
          </div>
        </div>

        <div className="info-item">
          <div className="info-icon">2</div>
          <div>
            <strong>Convert</strong>
            <p>Convert the handwriting into text.</p>
          </div>
        </div>

        <div className="info-item">
          <div className="info-icon">3</div>
          <div>
            <strong>Edit</strong>
            <p>Review and edit the result.</p>
          </div>
        </div>

        <div className="info-item">
          <div className="info-icon">4</div>
          <div>
            <strong>Copy</strong>
            <p>Use your digital text anywhere.</p>
          </div>
        </div>

      </section>


      {/* FOOTER */}
      <footer>
        <span>Handwriting Digitizer</span>
        <span>Convert · Edit · Copy</span>
      </footer>

    </div>
  );
}

export default App;