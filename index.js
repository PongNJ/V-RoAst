// Load environment variables from the google_key.env file
require('dotenv').config({ path: 'google_key.env' });

// Access the keys
const apiKey1 = process.env.GOOGLE_API_KEY_1;
const apiKey2 = process.env.GOOGLE_API_KEY_2;
const apiKey3 = process.env.GOOGLE_API_KEY_3;
const apiKey4 = process.env.GOOGLE_API_KEY_4;

// Function to choose an API key (this can be based on any logic you prefer)
function chooseApiKey() {
    // Example: Randomly choose one of the API keys
    const apiKeys = [apiKey1, apiKey2, apiKey3, apiKey4];
    const randomIndex = Math.floor(Math.random() * apiKeys.length);
    return apiKeys[randomIndex];
}

// Use the chosen API key
const selectedApiKey = chooseApiKey();
console.log('Selected API Key:', selectedApiKey);

// Example of using the selected API key with your existing configuration
generation_config = { 
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
    "api_key": selectedApiKey // Add the selected API key to the configuration
};