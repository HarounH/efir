import React, { useState } from 'react';

function TriangleSelector({selectorState, onClick}) {
  const [coordinates, setCoordinates] = useState({i: selectorState.i, j: selectorState.j, z: selectorState.k});
  const handleClick = (event) => {
    const rect = event.target.getBoundingClientRect();

    const y = (event.clientY - rect.top) / (rect.height);
    const x = (event.clientX - rect.left - (y * rect.width / 2)) / rect.width;
    const newSelectorState = {
      i: coordinates.i,
      j: coordinates.j,
      k: coordinates.k,
      x: x,
      y: y,
      z: 1.0 - x - y,
    };
    onClick(newSelectorState);
  };

  const handleChange = (event) => {
    const value = parseFloat(event.target.value);
    if (!isNaN(value)) {
      setCoordinates({
        ...coordinates,
        [event.target.name]: value,
      });
    }
  };

  return (
    <div style={{ position: 'relative' }}>
      <div
        onClick={handleClick}
        style={{
          width: 0,
          height: 0,
          borderStyle: 'solid',
          borderWidth: '0 300px 400px 300px',
          borderColor: `transparent transparent #007bff transparent`,
        }}
      />
      <div
        style={{
          position: 'absolute',
          left: 0,
          top: '100%',
          display: 'flex',
          justifyContent: 'space-between',
          width: '100%',
        }}
      >
        <input
          type="text"
          name="i"
          value={coordinates.i}
          onChange={handleChange}
          style={{ width: '33%', textAlign: "center"}}
        />
        <input
          type="text"
          name="j"
          value={coordinates.j}
          onChange={handleChange}
          style={{ width: '33%', textAlign: "center"}}
        />
        <input
          type="text"
          name="k"
          value={coordinates.k}
          onChange={handleChange}
          style={{ width: '33%', textAlign: "center"}}
        />
      </div>
    </div>
  );
};

export default TriangleSelector;
