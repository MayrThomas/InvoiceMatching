import numpy

from product import Product
import jellyfish
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def __monge_elkan_token_similarity(invoice_token: str, product_tokens: list[str]):
    """ Calculates the maximum similarity between the invoice token and all tokens of the product name.

        Similarity calculation can be done using any similarity or distance metric.
        This version will use Jaro-Winkler to determine similarity

    Parameters
    ----------
    :param invoice_token: token to calculate similarity for
    :param product_tokens: tokens of the product name
    :return: maximum similarity of invoice token with any product token: float
    """

    max_similarity = float("-inf")
    for token in product_tokens:
        similarity = jellyfish.jaro_winkler_similarity(invoice_token, token)

        if similarity > max_similarity:
            max_similarity = similarity

    return max_similarity


def levenshtein_matcher(reference_products: list[Product], invoice):
    """ Calculates the best possible match based on Levenshtein Distance

    Parameters
    ----------
    :param reference_products: products to match the invoice object against
    :param invoice: extracted line of invoice
    :return: product with the lowest Levenshtein Distance: tuple[Product, int]
    """

    shortest_distance = float("inf")
    match_product = None

    for product in reference_products:
        distance = jellyfish.levenshtein_distance(product.name, invoice.name)

        if distance < shortest_distance:
            shortest_distance = distance
            match_product = product

    return tuple((match_product, shortest_distance))


def jaro_matcher(reference_products: list[Product], invoice, use_prefix: bool = False):
    """ Calculates the Jaro similarity between the invoice and all products

    Parameters
    ----------
    :param reference_products: products to match the invoice object against
    :param invoice:  extracted line of invoice
    :param use_prefix: determines whether Jaro or Jaro-Winkler is used; default = False
    :return:  product with the highest similarity: tuple[Product, float]
    """

    best_match = float("-inf")
    match_product = Product("", "", "")

    for product in reference_products:
        if use_prefix:
            match = jellyfish.jaro_winkler_similarity(product.name, invoice.name)
        else:
            match = jellyfish.jaro_similarity(product.name, invoice.name)

        if match > best_match:
            best_match = match
            match_product = product

    return tuple((match_product, best_match))


def tfidf_matcher(reference_products: list[Product], invoice):
    """ Finds the most similar product based on TF-IDF and cosine similarity

    Parameters
    ----------
    :param reference_products: products to match the invoice object against
    :param invoice: extracted line of invoice
    :return: products with the highest similarity: tuple(list[Product], float)
    """

    documents = [invoice.name]
    documents.extend([x.name for x in reference_products])

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    all_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

    similarities = all_similarities[0:1, 1::1]
    max_similarity = numpy.amax(similarities)
    index = numpy.where(similarities == max_similarity)

    similar_products = [reference_products[x] for x in index[1]]

    return tuple((similar_products, max_similarity))


def jaccard_matcher(reference_products: list[Product], invoice):
    """ Finds the most similar product using Jaccard similarity

    Parameters
    ----------
    :param reference_products: products to match the invoice object against
    :param invoice: extracted line of invoice
    :return: product with the highest similarity: tuple(Product, float)
    """

    # tokenize all strings
    invoice_tokens = invoice.name.split()
    product_tokens = [x.name.split() for x in reference_products]

    max_similarity = float("-inf")
    match_product = None

    for index, product in enumerate(product_tokens):
        intersection = len(set(invoice_tokens).intersection(product))
        union = len(set(invoice_tokens).union(product))
        jaccard_similarity = float(intersection) / union

        if jaccard_similarity > max_similarity:
            max_similarity = jaccard_similarity
            match_product = reference_products[index]

    return tuple((match_product, max_similarity))


def monge_elkan_matcher(reference_products: list[Product], invoice):
    """ Finds the most similar product using Monge-Elkan strategy

        Monge-Elkan is a hybrid similarity metric that utilizes tokens and calculates the similarity of two strings
        using the similarity of the individual tokens.

    Parameters
    ----------
    :param reference_products: products to match the invoice object against
    :param invoice: extracted line of invoice
    :return: product with the highest similarity: tuple(Product, float)
    """

    # tokenize strings
    invoice_tokens = invoice.name.split()
    num_invoice_tokens = len(invoice_tokens)
    product_tokens = [x.name.split() for x in reference_products]

    max_similarity = float("-inf")
    match_product = None

    for index, product in enumerate(product_tokens):
        sum_token_similarity = 0.0

        for token in invoice_tokens:
            sum_token_similarity += __monge_elkan_token_similarity(token, product)

        monge_elkan = (1/num_invoice_tokens) * sum_token_similarity

        if monge_elkan > max_similarity:
            max_similarity = monge_elkan
            match_product = reference_products[index]

    return tuple((match_product, max_similarity))


    # TODO implement Soft TF-IDF
