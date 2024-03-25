import { useState } from 'react';
import { tokenizer_backend } from 'declarations/tokenizer_backend';
import { inference_backend } from 'declarations/inference_backend';

function App() {
  const [tokenIds, setTokenIds] = useState([]);

  function handleSubmit(event) {
    event.preventDefault();
    const text = event.target.elements.text.value;
    tokenizer_backend.tokenize_text(text)
      .then((tokens) => {
        setTokenIds(tokens);
      })
      .catch(error => {
        console.error("Error during tokenization:", error);
        setTokenIds([]);
      });
    return false;
  }

  return (
    <main style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', width: '100%', textAlign: 'center' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <img src="/logo2.svg" alt="DFINITY logo" style={{ height: '100px', width: 'auto', objectFit: 'contain', marginRight: '20px' }} />
        <img src="/Mod_Club.png" alt="ModClub logo" style={{ height: '100px', width: 'auto', objectFit: 'contain' }} />
      </div>
      <br /><br />
      <form action="#" onSubmit={handleSubmit}>
        <label htmlFor="text">Enter text: &nbsp;</label>
        <input id="text" type="text" />
        <button type="submit">Tokenize Text</button>
      </form>
      <section id="tokens">
        {tokenIds.length > 0 ? (
          <p>Token IDs: {tokenIds.join(', ')}</p>
        ) : (
          <p>No tokens to display.</p>
        )}
      </section>
    </main>
  );
}


export default App;