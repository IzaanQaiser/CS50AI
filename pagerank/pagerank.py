import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    
    probabilities = {}
    outgoing_links = corpus[page]

    if not outgoing_links:
        equal_prob = 1 / len(corpus)
        for p in corpus:
            probabilities[p] = equal_prob
    
    else:
        base_prob = (1 - damping_factor) / len(corpus)
        for p in corpus:
            probabilities[p] = base_prob

        link_prob = damping_factor / len(outgoing_links)
        for linked_page in outgoing_links:
            probabilities[linked_page] += link_prob
    
    return probabilities


def sample_pagerank(corpus, damping_factor, n):
    import random

    page_rank = {page: 0 for page in corpus}
    current_page = random.choice(list(corpus.keys()))

    for _ in range(n):
        page_rank[current_page] += 1
        transition_probabilities = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(
            population=list(transition_probabilities.keys()),
            weights=transition_probabilities.values(),
            k=1
        )[0]

    total_samples = sum(page_rank.values())
    for page in page_rank:
        page_rank[page] /= total_samples

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    num_pages = len(corpus)
    base_rank = (1 - damping_factor) / num_pages
    ranks = {page: 1 / num_pages for page in corpus}
    updated_ranks = {}
    converged = False

    while not converged:
        for target_page in ranks:
            rank_sum = 0
            for source_page, links in corpus.items():
                if target_page in links:
                    rank_sum += ranks[source_page] / len(links)
                if not links:
                    rank_sum += ranks[source_page] / num_pages
            updated_ranks[target_page] = base_rank + damping_factor * rank_sum

        converged = all(
            abs(ranks[page] - updated_ranks[page]) <= 0.001 for page in ranks
        )
        ranks.update(updated_ranks)

    return ranks



if __name__ == "__main__":
    main()
