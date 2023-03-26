# Genlock Showcase
Welcome to the Genlock Showcase! This is a web page built with React that demonstrait the usage of our service. It includes a Unity WebGL game that allows users to generate and receive NFTs of gaming assets. The web page also uses TON Connect 2.0 to connect a TON wallet and to sign transactions.

<b>This is just an example of what can be done with Genlock. The Genlock Showcase is a simple game that calls our API to demonstrate the functionality of Genlock.</b>

## How to Run
Install dependencies: `yarn`
Start the development server: `yarn start`

## Unity Game
The Unity game build is included in this project and can be found [here](https://github.com/tema7707/Genlock/tree/main/Showcase/public/build). We used the [Unity WebGL](https://react-unity-webgl.dev/) library to run Unity inside the React App.

## TON Connect 2.0
To enter and sign transactions, this project uses TON Connect 2.0. The usage of TON Connect 2.0 can be found in this [file](https://github.com/tema7707/Genlock/blob/main/Showcase/src/components/TxForm/TxForm.tsx).

## Credits
This project was based on the [demo-dapp repository by ton-connect](https://github.com/ton-connect/demo-dapp). We would like to thank ton-connect for their contributions to the open source community.
