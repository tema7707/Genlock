import React, { useEffect } from 'react';
import { AppTitle } from 'src/components/AppTitle/AppTitle';
import { AuthButton } from 'src/components/AuthButton/AuthButton';
import { TxForm } from 'src/components/TxForm/TxForm';
import { connector } from 'src/connector';
import './app.scss';

function App() {
	useEffect(() => {
		connector.restoreConnection();
	}, []);

	return (
		<div className="app">
			<header>
				<AppTitle />
				<AuthButton />
			</header>
			<main>
				<TxForm />
			</main>
		</div>
	);
}

export default App;
