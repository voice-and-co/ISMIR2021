

import os
from extractor import melody_extractor, rhythm_extractor, harmony_extractor, contour_extractor, onsets_extractor
from extractor.pattern_coincidence import pattern_coincidence

if __name__ == "__main__":
    if not os.path.exists('../temp'):
        os.mkdir('../temp')

    melody = False
    rhythm = False
    harmony = False
    contours = False
    onset = True
    example = False

    # Run melody and rhythm extractors
    if rhythm or melody:
        pattern_sizes = [1, 2, 3, 4, 5, 6, 7]
        corpora = ['classical', 'modern']
        for corpus in corpora:
            for size in pattern_sizes:
                if rhythm:
                    rhythm_extractor.run_extractor(corpus, size)
                if melody:
                    melody_extractor.run_extractor(corpus, size)
        if rhythm:
            pattern_coincidence('rhythm', pattern_sizes, 'classical')
            pattern_coincidence('rhythm', pattern_sizes, 'modern')
        if melody:
            pattern_coincidence('melody', pattern_sizes, 'classical')
            pattern_coincidence('melody', pattern_sizes, 'modern')

    # Run harmony extractors
    if harmony:
        harmony_extractor.run_extractor(corpus='classical', type_descriptor="cosine similarity")
        harmony_extractor.run_extractor(corpus='modern', type_descriptor="cosine similarity")

    if contours:
        contour_extractor.run_extractor(corpus='classical')
        contour_extractor.run_extractor(corpus='modern')

    if onset:
        onsets_extractor.run_extractor(corpus='classical')
        onsets_extractor.run_extractor(corpus='modern')

    if example:
        pattern_sizes = [1, 2, 3, 4, 5, 6, 7]

        for size in pattern_sizes:
            rhythm_extractor.run_extractor("example", size)
            melody_extractor.run_extractor("example", size)
        pattern_coincidence("rhythm", pattern_sizes, 'example')
        pattern_coincidence("melody", pattern_sizes, 'example')

        harmony_extractor.run_extractor(corpus='example', type_descriptor="cosine similarity")
        contour_extractor.run_extractor(corpus='example')
        onsets_extractor.run_extractor(corpus='example')
