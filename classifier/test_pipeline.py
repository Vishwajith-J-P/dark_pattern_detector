import asyncio
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.playwright_scraper import scrape
from scraper.html_parser import parse_html
from classifier.predict import (
    predict_elements,
    calculate_manipulation_score,
    build_explainability_cards
)

async def test():
    print("Scraping...")
    result = await scrape('https://www.flipkart.com/apple-iphone-15/p/itmbf14ef54f645d')

    if 'error' in result:
        print('Scrape error:', result['error'])
        return

    print("Parsing elements...")
    elements = parse_html(result['html'], result.get('elements', []))
    print(f"Total elements: {len(elements)}")

    print("Running classifier...")
    detections = predict_elements(elements)
    print(f"Elements with dark patterns: {len(detections)}")

    print("\nCalculating manipulation score...")
    score, compounds, per_pattern = calculate_manipulation_score(detections)
    print(f"Manipulation Score: {score}/100")

    if compounds:
        print("\nCompound patterns detected:")
        for c in compounds:
            print(f"  ⚠ {c['name']} (boost: {c['boost']}x)")

    print("\nPer-pattern breakdown:")
    for label, s in sorted(per_pattern.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label:35} {s:.3f}")

    print("\nExplainability cards:")
    cards = build_explainability_cards(detections)
    for card in cards:
        print(f"\n  [{card['display_name']}] — confidence: {card['confidence']}")
        print(f"  Bias exploited: {card['cognitive_bias']}")
        print(f"  Example: {card['example_text'][:80]}")

asyncio.run(test())