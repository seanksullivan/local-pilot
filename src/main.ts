import './styles/app.css';
import { setupOllamaChat } from './app/ollama-chat';

const app = document.getElementById('app');
if (!app) {
  throw new Error('#app root element not found');
}

void setupOllamaChat(app);
