@app.route('/api/analyse-elements', methods=['POST'])
def analyse_elements():
    data     = request.get_json()
    elements = data.get('elements', [])
    url      = data.get('url', 'unknown')

    if not elements:
        return jsonify({'error': 'No elements provided'}), 400

    # Convert to format predict_elements expects
    formatted = [
        {
            'text':           e.get('text', ''),
            'tag':            e.get('tag', 'div'),
            'bbox':           e.get('bbox'),
            'heuristic_hits': [],
            'priority':       False
        }
        for e in elements
        if e.get('text', '').strip()
    ]

    detections = predict_elements(formatted)
    score, compounds, per_pattern = calculate_manipulation_score(detections)
    cards = build_explainability_cards(detections)

    domain = urlparse(url).netloc.replace('www.', '')
    if domain:
        save_result(url, domain, score, list(per_pattern.keys()))

    return jsonify({
        'score':            score,
        'compounds':        compounds,
        'per_pattern':      per_pattern,
        'cards':            cards,
        'detections':       [
            {
                'text':     d['text'][:100],
                'bbox':     d.get('bbox'),
                'patterns': d['detected_patterns'],
            }
            for d in detections
        ],
        'total_detections': len(detections),
        'domain':           domain,
    })