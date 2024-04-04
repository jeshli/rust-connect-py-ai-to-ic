import { useState, useMemo, useEffect } from 'react';
import { tokenizer_backend } from 'declarations/tokenizer_backend';
import { inference_backend } from 'declarations/inference_backend';



function App() {
  const [tokenIds, setTokenIds] = useState([]);
  const [tokenValues, setTokenValues] = useState([]);
  const [inferenceResults, setInferenceResults] = useState([]);
  const [inferenceResultWeights, setInferenceResultWeights] = useState([]);
  const [inferenceExpResults, setInferenceExpResults] = useState([]);
  const [bias, setBias] = useState(-3); // Default bias value set to 2
  const [scale, setScale] = useState(8); // Default bias value set to 2

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
        //const transformedScores = scores.map(score => Math.exp(scale * (score + bias)) );
        const transformedScores = scores.map(score => Math.exp(scale * score + bias) );
        setInferenceExpResults(transformedScores);
        const scoreWeights = scores.map(score => Math.max(Math.exp(score),1) + 1 );
        setInferenceResultWeights(scoreWeights);
      })
      .catch(error => {
        console.error("Error during inference:", error);
        setInferenceResults([]);
        setInferenceExpResults([]);
      });

    return false;
  }


  // Calculate the sum of inference results
  const totalScore = useMemo(() => {
      // Ensure that inferenceResults and inferenceResultWeights are defined and have the same length
      if (!inferenceResults || !inferenceResultWeights || inferenceResults.length !== inferenceResultWeights.length) {
        console.error('inferenceResults and inferenceResultWeights must be defined and have the same length.');
        return 0;
      }

      // Calculate the weighted sum
      return inferenceResults.reduce((acc, curr, index) => acc + curr * inferenceResultWeights[index], 0);
  }, [inferenceResults, inferenceResultWeights]); // Recalculate if either array changes

  useEffect(() => {
    //const transformedScores = inferenceResults.map(score => Math.exp(scale * (score + bias)));
    const transformedScores = inferenceResults.map(score => Math.exp(scale * score + bias));
    setInferenceExpResults(transformedScores);
  }, [inferenceResults, scale, bias]);

  return (
    <main style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', width: '100%', textAlign: 'center' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <img src="/logo2.svg" alt="DFINITY logo" style={{ height: '100px', width: 'auto', objectFit: 'contain', marginRight: '20px' }} />
        <img src="/DecideAI.png" alt="DecideAI logo" style={{ height: '100px', width: 'auto', objectFit: 'contain' }} />
      </div>
      <br /><br />
      <h2>Redactor v0.1</h2>
      <form action="#" onSubmit={handleSubmit}>
        <label htmlFor="text">Enter text: &nbsp;</label>
        <input id="text" type="text" />
        <button type="submit">Tokenize and Evaluate Text</button>
      </form>
      <section id="tokens">
        {tokenValues.length > 0 ? (
          <p>Tokens:&nbsp;
            {tokenValues.map((token, index) => (
              <span key={index} style={{
                //backgroundColor: getRandomColor(),
                filter: `blur(${inferenceExpResults[index]}px)`, // Apply dynamic blurring based on inference result
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

        <p>Total Score: {totalScore.toFixed(2)}</p>
      </section>
      <section class="customize-section">
        <p>Customize your app or let users personalize their settings.</p>
        <div>
          <label htmlFor="biasSlider">Adjust Bias: </label>
          <input
            id="biasSlider"
            type="range"
            min="-5" // Minimum value of bias
            max="5" // Maximum value of bias
            value={bias}
            onChange={(e) => setBias(Number(e.target.value))}
            step="0.01" // Adjust for finer control
          />
          <span>{bias}</span> {/* Display the current bias value */}
        </div>

        <div>
          <label htmlFor="scaleSlider">Adjust Scale: </label>
          <input
            id="scaleSlider"
            type="range"
            min="5" // Minimum value of bias
            max="15" // Maximum value of bias
            value={scale}
            onChange={(e) => setScale(Number(e.target.value))}
            step="0.01" // Adjust for finer control
          />
          <span>{scale}</span> {/* Display the current bias value */}
        </div>
      </section>

      {/*
      <section id="inferenceResults">
        {inferenceResults.length > 0 ? (
          <p>Inference Results: {inferenceResults.join(', ')}</p>
        ) : (
          <p>No score to display.</p>
        )}
      </section>
      */}
      <section class="note">
        <br /><br />
        <p>This is a secure, permanent, trustless, and free API.</p>
      </section>

      <section class="api-section">
        <div class="api-container">
            <h3>Tokenization</h3>
            <div class="api-details">
                <p><strong>Canister:</strong>hgczj-kiaaa-aaaao-a3ksq-cai</p>
                <p><strong>Query:</strong> tokenize_text: (text) → (vec nat64, vec text) query</p>
            </div>
        </div>
        <div class="api-container">
            <h3>Inference</h3>
            <div class="api-details">
                <p><strong>Canister:</strong>hbd75-hqaaa-aaaao-a3ksa-cai</p>
                <p><strong>Query:</strong> model_inference: (vec int64) → (vec float32) query</p>
            </div>
        </div>
      </section>


    </main>
  );
}


export default App;