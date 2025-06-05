import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_closest_face(new_embedding, known_faces, threshold=0.4):
    """
    :param new_embedding: np.ndarray — эмбеддинг лица (1D)
    :param known_faces: list — [(name, role, group_name, embedding), ...]
    :param threshold: float — минимальная схожесть
    :return: tuple | None — (name, similarity, role, group) или None
    """

    new_embedding = np.array(new_embedding, dtype=np.float32).reshape(1, -1)

    best_match = None
    best_similarity = -1
    best_group = ""
    best_role = ""

    for name, role, group_name, embedding in known_faces:
        db_embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)

        similarity = cosine_similarity(db_embedding, new_embedding)[0][0]

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = name
            best_group = group_name
            best_role = role

    if best_similarity > threshold:
        return best_match, best_similarity, best_role, best_group
    return None
