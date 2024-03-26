import { useState } from 'react';
import { tokenizer_backend } from 'declarations/tokenizer_backend';
import { inference_backend } from 'declarations/inference_backend';

function getRandomColor() {
  const letters = '0123456789ABCDEF';
  let color = '#';
  for (let i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)];
  }
  return color;
}

function App() {
  const [tokenIds, setTokenIds] = useState([]);
  const [tokenValues, setTokenValues] = useState([]);
  const [inferenceResults, setInferenceResults] = useState([]);

  function handleSubmit(event) {
    event.preventDefault();
    const text = event.target.elements.text.value;
    tokenizer_backend.tokenize_text(text)
      .then(([token_ids, token_values]) => {
        setTokenIds(token_ids);
        setTokenValues(token_values);
        return inference_backend.model_inference(token_ids);
      })
      .catch(error => {
        console.error("Error during tokenization:", error);
        setTokenIds([]);
        setTokenValues([]);
      }).then((scores) => {
        setInferenceResults(scores);
      })
      .catch(error => {
        console.error("Error during inference:", error);
        setInferenceResults([]);
      });

    return false;
  }

  return (
    <main style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', width: '100%', textAlign: 'center' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <img src="/logo2.svg" alt="DFINITY logo" style={{ height: '100px', width: 'auto', objectFit: 'contain', marginRight: '20px' }} />
        <img src="/DecideAI.png" alt="DecideAI logo" style={{ height: '100px', width: 'auto', objectFit: 'contain' }} />
      </div>
      <br /><br />
      <form action="#" onSubmit={handleSubmit}>
        <label htmlFor="text">Enter text: &nbsp;</label>
        <input id="text" type="text" />
        <button type="submit">Tokenize Text</button>
      </form>
      <section id="tokens">
        {tokenValues.length > 0 ? (
          <p>Tokens:&nbsp;
            {tokenValues.map((token, index) => (
              <span key={index} style={{
                backgroundColor: getRandomColor(),
                margin: '0', // Removed space between spans
                padding: '0'  // Adjust according to preference
              }}>
                {token}
              </span>
            ))}
          </p>
        ) : (
          <p>No tokens to display.</p>
        )}
      </section>
      <section id="inferenceResults">
        {inferenceResults.length > 0 ? (
          <p>Inference Results: {inferenceResults.join(', ')}</p>
        ) : (
          <p>No score to display.</p>
        )}
      </section>
    </main>
  );
}


export default App;