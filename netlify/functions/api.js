// Proxy para football-data.org — evita CORS en el browser
exports.handler = async (event) => {
  const apiKey = event.headers['x-auth-token'] || '';
  const path = event.path.replace(/^\/.netlify\/functions\/api/, '');
  const qs = event.rawQuery ? '?' + event.rawQuery : '';
  const url = `https://api.football-data.org/v4${path}${qs}`;

  try {
    const resp = await fetch(url, {
      headers: { 'X-Auth-Token': apiKey }
    });
    const data = await resp.text();
    return {
      statusCode: resp.status,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      },
      body: data,
    };
  } catch (err) {
    return {
      statusCode: 502,
      body: JSON.stringify({ error: err.message }),
    };
  }
};
