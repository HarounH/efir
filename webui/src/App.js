import React, { useState } from 'react';
import "./App.css";
import TriangleSelector from "./Triangle";
import MnistImage from "./MnistImage";

function App() {
  const [selectorState, setSelectorState] = useState({
    i: 0,
    j: 0,
    k: 0,
    x: 0.0,
    y: 0.0,
    z: 1.0,
  });
  const [url, setUrl] = useState('deadpath');

  async function handleChange(selectorState) {
    setSelectorState(selectorState);
    const x = selectorState.x;
    const y = selectorState.y;
    const z = selectorState.z;
    const i = selectorState.i;
    const j = selectorState.j;
    const k = selectorState.k;
    // call service
    const query = `http://127.0.0.1:8000/forward?i=${i}&j=${j}&k=${k}&x=${x}&y=${y}&z=${z}`;
    console.log(query);
    const server_response = await fetch(query);
    const blob = await server_response.blob();
    const new_url = URL.createObjectURL(blob);
    console.log(new_url);
    setUrl(new_url);
  };
  return (
    <div className="app-container">
      <div className="triangle-panel">
        <TriangleSelector selectorState={selectorState} onClick={(selectorState) => handleChange(selectorState)} />
      </div>
      <div className="mnist-image-panel">
        <MnistImage url={url} />
      </div>
    </div>
  );
}

export default App;
