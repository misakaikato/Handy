//! Pure correction-learning logic: atomization, region location, diffing
//! and replacement acceptance. Platform independent and unit-tested on
//! every platform; only macOS wires it to the Accessibility reader in the
//! parent module.

/// Pastes longer than this many word atoms are not analyzed; the LCS diff
/// grows quadratically and long dictations rarely need per-term learning.
const MAX_PASTED_WORD_ATOMS: usize = 600;
/// Upper bound on the atoms of the field region we are willing to diff.
const MAX_REGION_ATOMS: usize = 2_400;
/// Guard against pathological field sizes (huge documents in AXValue).
const MAX_FIELD_BYTES: usize = 1_000_000;
/// A paste with more replaced spans than this looks like a rewrite.
const MAX_REPLACEMENT_HUNKS: usize = 8;
/// Share of the pasted word atoms that may be replaced before the whole
/// correction is considered a rewrite and ignored.
const MAX_REPLACED_FRACTION: f64 = 0.4;
/// Absolute floor under the fraction guard so short pastes can still fix a
/// few terms.
const MAX_REPLACED_ATOMS_FLOOR: usize = 4;
/// Length of the text anchors used to locate the pasted region.
const PREFIX_ANCHOR_CHARS: usize = 16;
const SUFFIX_ANCHOR_CHARS: usize = 8;

use crate::audio_toolkit::text::{canonical_syllable_of, is_han};

/// Splits text into diff atoms: maximal ASCII word runs, single Han
/// characters, and every other character as its own atom. Atomization is
/// lossless — concatenating the atoms reproduces the input.
fn atomize(text: &str) -> Vec<String> {
    let mut atoms = Vec::new();
    let mut latin = String::new();
    for c in text.chars() {
        if is_latin_word_char(c) {
            latin.push(c);
        } else {
            if !latin.is_empty() {
                atoms.push(std::mem::take(&mut latin));
            }
            atoms.push(c.to_string());
        }
    }
    if !latin.is_empty() {
        atoms.push(latin);
    }
    atoms
}

fn is_latin_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '-' | '\'' | '&' | '.' | '/')
}

/// Word atoms participate in correction matching; separator atoms (spaces,
/// punctuation) only structure the diff.
fn is_word_atom(atom: &str) -> bool {
    atom.chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphanumeric() || is_han(c))
}

fn is_han_str(text: &str) -> bool {
    !text.is_empty() && text.chars().all(is_han)
}

/// Trims non-alphanumeric edges off an atom, e.g. "chatgpt." -> "chatgpt".
/// Returns `None` when nothing alphanumeric remains.
fn atom_core(atom: &str) -> Option<&str> {
    let core = atom.trim_matches(|c: char| !c.is_alphanumeric());
    (!core.is_empty()).then_some(core)
}

/// Rounds a byte offset down to a char boundary of `s`.
fn floor_char_boundary(s: &str, byte: usize) -> usize {
    let mut idx = byte.min(s.len());
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Byte offset of the start of the character `chars` positions before
/// `byte` (which must already be a char boundary or is floored first).
fn back_chars(s: &str, byte: usize, chars: usize) -> usize {
    let mut idx = floor_char_boundary(s, byte);
    for _ in 0..chars {
        if idx == 0 {
            break;
        }
        idx = floor_char_boundary(s, idx - 1);
    }
    idx
}

/// Byte offset of the end of the character `chars` positions after `byte`.
fn forward_chars(s: &str, byte: usize, chars: usize) -> usize {
    let mut idx = floor_char_boundary(s, byte);
    for _ in 0..chars {
        match s[idx..].chars().next() {
            Some(c) => idx += c.len_utf8(),
            None => break,
        }
    }
    idx
}

/// Locates the (possibly edited) pasted text inside the field value and
/// returns that region as a slice of `field`.
///
/// The region is anchored with an exact prefix and suffix of the paste, so a
/// correction inside the paste still resolves. When an anchor itself was
/// edited away, the anchor falls back to the first/last word atom, and
/// finally to a bounded extension from the opposite anchor. Returns `None`
/// when the region cannot be located with confidence.
fn locate_region<'f>(pasted: &str, field: &'f str) -> Option<&'f str> {
    let pasted_chars = pasted.chars().count();
    if pasted_chars == 0 || field.is_empty() {
        return None;
    }

    let prefix: String = pasted.chars().take(PREFIX_ANCHOR_CHARS).collect();
    let suffix: String = pasted
        .chars()
        .rev()
        .take(SUFFIX_ANCHOR_CHARS)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    // Prefer the last occurrence of each anchor: repeated pastes append.
    let prefix_hit = field.rfind(&prefix);
    let suffix_end = field.rfind(&suffix).map(|byte| byte + suffix.len());

    let atoms = atomize(pasted);
    let words: Vec<&str> = atoms
        .iter()
        .filter(|atom| is_word_atom(atom.as_str()))
        .filter_map(|atom| atom_core(atom.as_str()))
        .collect();

    let mut start = prefix_hit;
    let mut end = suffix_end;

    if start.is_none() {
        // The correction touched the head of the paste. Try the first word.
        start = words.first().and_then(|w| field.rfind(w));
    }
    if end.is_none() {
        // The correction touched the tail of the paste. Try the last word.
        end = words
            .last()
            .and_then(|w| field.rfind(w).map(|byte| byte + w.len()));
    }

    // One side still missing: extend from the other side by a bounded factor
    // of the paste length, accepting some surrounding text into the diff.
    if end.is_none() {
        let start = start?;
        end = Some(forward_chars(field, start, pasted_chars * 2));
    }
    if start.is_none() {
        let end = end?;
        start = Some(back_chars(field, end, pasted_chars));
    }

    let (start, end) = (start?, end?);
    if end <= start {
        return None;
    }

    let region_chars = field[start..end].chars().count();
    let lower = (pasted_chars as f64 * 0.4) as usize;
    let upper = (pasted_chars as f64 * 3.0).ceil() as usize;
    if region_chars < lower || region_chars > upper {
        return None;
    }

    Some(&field[start..end])
}

/// One non-equal hunk of the LCS diff, as index ranges into the atom
/// slices. Han hunks are turned into term corrections by windowing around
/// the changed characters (see [`learn_han_term`]).
struct Hunk {
    old: std::ops::Range<usize>,
    new: std::ops::Range<usize>,
}

/// Classic LCS diff over atoms. Returns the non-equal hunks in document
/// order, as index ranges into `old`/`new`.
fn diff_hunks(old: &[String], new: &[String]) -> Vec<Hunk> {
    let (n, m) = (old.len(), new.len());
    let width = m + 1;
    // LCS-length table; u32 is plenty for the atom caps enforced by callers.
    let mut table = vec![0u32; (n + 1) * width];

    for i in (0..n).rev() {
        for j in (0..m).rev() {
            table[i * width + j] = if old[i] == new[j] {
                table[(i + 1) * width + j + 1] + 1
            } else {
                table[(i + 1) * width + j].max(table[i * width + j + 1])
            };
        }
    }

    let mut hunks = Vec::new();
    let mut cur_old: Option<usize> = None;
    let mut cur_new: Option<usize> = None;
    let (mut i, mut j) = (0, 0);

    let flush = |cur_old: &mut Option<usize>,
                 cur_new: &mut Option<usize>,
                 i: usize,
                 j: usize,
                 hunks: &mut Vec<Hunk>| {
        if let (Some(old_start), Some(new_start)) = (*cur_old, *cur_new) {
            hunks.push(Hunk {
                old: old_start..i,
                new: new_start..j,
            });
        }
        *cur_old = None;
        *cur_new = None;
    };

    while i < n && j < m {
        if old[i] == new[j] {
            flush(&mut cur_old, &mut cur_new, i, j, &mut hunks);
            i += 1;
            j += 1;
        } else if table[(i + 1) * width + j] >= table[i * width + j + 1] {
            cur_old.get_or_insert(i);
            i += 1;
        } else {
            cur_new.get_or_insert(j);
            j += 1;
        }
    }
    flush(&mut cur_old, &mut cur_new, n, m, &mut hunks);

    hunks
}

/// Absorbs equal Han neighbors into a hunk whose sides are pure Han.
/// Character-level diffs surface a homophone fix as a single changed
/// character (威 -> 微); the actual corrected term is that character plus
/// some unchanged Han context. Every window around the changed characters
/// whose toneless pinyin matches is a candidate; the shortest window wins
/// (ties prefer extending right), yielding 微信 from 威 -> 微 inside 威信.
fn learn_han_term(old: &[String], new: &[String], hunk: &Hunk) -> Option<String> {
    if !pasted_atoms_all_han(old, &hunk.old) || !pasted_atoms_all_han(new, &hunk.new) {
        return None;
    }

    // Maximal Han run around the hunk on both sides.
    let mut old_run_start = hunk.old.start;
    while old_run_start > 0 && is_han_str(&old[old_run_start - 1]) {
        old_run_start -= 1;
    }
    let mut old_run_end = hunk.old.end;
    while old_run_end < old.len() && is_han_str(&old[old_run_end]) {
        old_run_end += 1;
    }
    let mut new_run_start = hunk.new.start;
    while new_run_start > 0 && is_han_str(&new[new_run_start - 1]) {
        new_run_start -= 1;
    }
    let mut new_run_end = hunk.new.end;
    while new_run_end < new.len() && is_han_str(&new[new_run_end]) {
        new_run_end += 1;
    }

    // Extension is bounded by the run on both slices; extension atoms are
    // diff-equal, so old/new window indices stay aligned.
    let max_left = (hunk.old.start - old_run_start).min(hunk.new.start - new_run_start);
    let max_right = (old_run_end - hunk.old.end).min(new_run_end - hunk.new.end);

    let mut best: Option<(usize, usize, String)> = None; // (total len, left ext, term)
    for left in 0..=max_left {
        for right in 0..=max_right {
            let old_window: String = old[hunk.old.start - left..hunk.old.end + right].concat();
            let new_window: String = new[hunk.new.start - left..hunk.new.end + right].concat();
            let (old_len, new_len) = (old_window.chars().count(), new_window.chars().count());
            if !(2..=8).contains(&old_len) || !(2..=8).contains(&new_len) {
                continue;
            }
            if !han_pinyin_matches(&old_window, &new_window) {
                continue;
            }
            let total = old_len + new_len;
            let better = best
                .as_ref()
                .is_none_or(|(best_total, best_left, _)| (total, left) < (*best_total, *best_left));
            if better {
                best = Some((total, left, new_window));
            }
        }
    }
    best.map(|(_, _, term)| term)
}

/// Whether every atom in `atoms[range]` is a Han character.
fn pasted_atoms_all_han(atoms: &[String], range: &std::ops::Range<usize>) -> bool {
    !range.is_empty() && atoms[range.clone()].iter().all(|atom| is_han_str(atom))
}

/// Toneless-pinyin similarity for equal-length Han spans: identical
/// canonical syllables, or at most one mismatched syllable in spans of three
/// or more characters.
fn han_pinyin_matches(old: &str, new: &str) -> bool {
    let syllables =
        |word: &str| -> Option<Vec<String>> { word.chars().map(canonical_syllable_of).collect() };
    let (Some(old_syllables), Some(new_syllables)) = (syllables(old), syllables(new)) else {
        return false;
    };
    let mismatches = old_syllables
        .iter()
        .zip(&new_syllables)
        .filter(|(a, b)| a != b)
        .count();
    mismatches == 0 || (new.chars().count() >= 3 && mismatches <= 1)
}

/// Extracts words worth learning from one correction round.
///
/// `pasted` is what Handy pasted, `field` the focused field's value read
/// back later, `existing` the current custom words (used for dedupe).
pub(crate) fn extract_learned_words(pasted: &str, field: &str, existing: &[String]) -> Vec<String> {
    let pasted = pasted.trim();
    if pasted.is_empty() || field.is_empty() || field.len() > MAX_FIELD_BYTES {
        return Vec::new();
    }

    let pasted_atoms = atomize(pasted);
    let pasted_word_count = pasted_atoms
        .iter()
        .filter(|atom| is_word_atom(atom.as_str()))
        .count();
    if pasted_word_count == 0 || pasted_word_count > MAX_PASTED_WORD_ATOMS {
        return Vec::new();
    }

    // The paste landed unchanged: nothing to learn.
    if field.contains(pasted) {
        return Vec::new();
    }

    let Some(region) = locate_region(pasted, field) else {
        return Vec::new();
    };
    let region_atoms = atomize(region);
    if region_atoms.len() > MAX_REGION_ATOMS {
        return Vec::new();
    }

    // Only replacement hunks (word atoms on both sides) are corrections of a
    // term; pure insertions and deletions are ignored.
    let replacements: Vec<Hunk> = diff_hunks(&pasted_atoms, &region_atoms)
        .into_iter()
        .filter(|hunk| {
            (pasted_atoms_all_han(&pasted_atoms, &hunk.old)
                && pasted_atoms_all_han(&region_atoms, &hunk.new))
                || (hunk_word_count(&pasted_atoms, &hunk.old) > 0
                    && hunk_word_count(&region_atoms, &hunk.new) > 0)
        })
        .collect();
    if replacements.is_empty() || replacements.len() > MAX_REPLACEMENT_HUNKS {
        return Vec::new();
    }

    // Ignore large rewrites: a correction worth learning touches a small
    // part of the paste.
    let replaced_atoms: usize = replacements
        .iter()
        .map(|hunk| hunk_word_count(&pasted_atoms, &hunk.old))
        .sum();
    let allowed = ((MAX_REPLACED_ATOMS_FLOOR as f64)
        .max(MAX_REPLACED_FRACTION * pasted_word_count as f64)) as usize;
    if replaced_atoms > allowed {
        return Vec::new();
    }

    let mut learned: Vec<String> = Vec::new();
    for hunk in replacements {
        let word = if pasted_atoms_all_han(&pasted_atoms, &hunk.old)
            && pasted_atoms_all_han(&region_atoms, &hunk.new)
        {
            learn_han_term(&pasted_atoms, &region_atoms, &hunk)
        } else {
            accept_replacement(&pasted_atoms[hunk.old], &region_atoms[hunk.new])
        };
        let Some(word) = word else {
            continue;
        };
        if !same_word_in(&word, existing) && !same_word_in(&word, &learned) {
            learned.push(word);
        }
    }
    learned
}

/// Number of word atoms in one side of a hunk.
fn hunk_word_count(atoms: &[String], range: &std::ops::Range<usize>) -> usize {
    atoms[range.clone()]
        .iter()
        .filter(|atom| is_word_atom(atom.as_str()))
        .count()
}

/// Decides whether one replacement hunk is a term correction worth learning
/// and returns the word to learn (the corrected side). Pure-Han hunks are
/// handled by [`learn_han_term`]; everything else goes through the Latin
/// term checks (mixed-script spans never pass them).
fn accept_replacement(old: &[String], new: &[String]) -> Option<String> {
    let old_cores: Vec<&str> = old
        .iter()
        .filter(|atom| is_word_atom(atom.as_str()))
        .filter_map(|atom| atom_core(atom.as_str()))
        .collect();
    let new_cores: Vec<&str> = new
        .iter()
        .filter(|atom| is_word_atom(atom.as_str()))
        .filter_map(|atom| atom_core(atom.as_str()))
        .collect();
    if old_cores.is_empty() || new_cores.is_empty() {
        return None;
    }

    accept_latin_replacement(&old_cores, &new_cores)
}

/// Latin side: the corrected text must be a single term; the pasted side may
/// span up to three spoken words ("charge bee" -> "ChargeBee"). Accepts on
/// edit-distance/Soundex similarity, mirroring the custom-word matcher.
fn accept_latin_replacement(old: &[&str], new: &[&str]) -> Option<String> {
    if new.len() != 1 || old.is_empty() || old.len() > 3 {
        return None;
    }
    let new_word = new[0];
    if !is_latin_token(new_word) || !(2..=24).contains(&new_word.chars().count()) {
        return None;
    }
    if !old
        .iter()
        .all(|w| is_latin_token(w) && w.chars().count() <= 24)
    {
        return None;
    }

    let old_joined = old.concat();
    if old_joined.eq_ignore_ascii_case(new_word) {
        // Pure case fix: learn it only when more than the leading capital
        // changed ("Chargebee" -> "ChargeBee"), never a sentence-start
        // capitalization fix ("hello" -> "Hello").
        return (!differs_only_by_leading_case(&old_joined, new_word))
            .then(|| new_word.to_string());
    }

    let old_len = old_joined.chars().count();
    let new_len = new_word.chars().count();
    let max_len = old_len.max(new_len);
    // Same guard as the custom-word matcher: at most 25% (min 2 chars) length difference.
    if (old_len.abs_diff(new_len)) as f64 > (max_len as f64 * 0.25).max(2.0) {
        return None;
    }

    let old_lower = old_joined.to_ascii_lowercase();
    let new_lower = new_word.to_ascii_lowercase();
    let levenshtein_score = strsim::levenshtein(&old_lower, &new_lower) as f64 / max_len as f64;

    // Soundex is only meaningful for purely alphabetic terms.
    let phonetic = old_lower.chars().all(|c| c.is_ascii_alphabetic())
        && new_lower.chars().all(|c| c.is_ascii_alphabetic())
        && natural::phonetics::soundex(&old_lower, &new_lower);

    if levenshtein_score <= 0.2 || (phonetic && levenshtein_score <= 0.5) {
        Some(new_word.to_string())
    } else {
        None
    }
}

/// ASCII term check: alphanumeric, optionally with inner `-`, `'`, `&`, `.`,
/// `/`, and never a bare number.
fn is_latin_token(token: &str) -> bool {
    token
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '\'' | '&' | '.' | '/'))
        && token.chars().any(|c| c.is_ascii_alphabetic())
}

/// Whether two ASCII strings (case-insensitively equal) differ — at all,
/// and only in the case of their first character.
fn differs_only_by_leading_case(old: &str, new: &str) -> bool {
    if old == new {
        return false;
    }
    let (old_bytes, new_bytes) = (old.as_bytes(), new.as_bytes());
    old_bytes.len() == new_bytes.len()
        && old_bytes[1..] == new_bytes[1..]
        && old_bytes
            .first()
            .zip(new_bytes.first())
            .is_some_and(|(a, b)| a.eq_ignore_ascii_case(b))
}

/// Dedupe key: case-insensitive for Latin words, exact for Han-containing
/// words.
fn same_word_in(word: &str, list: &[String]) -> bool {
    let contains_han = word.chars().any(is_han);
    list.iter().any(|existing| {
        if contains_han {
            existing == word
        } else {
            existing.trim().eq_ignore_ascii_case(word.trim())
        }
    })
}

/// Removes the words of one learn event from the custom-word list, using the
/// same match rule as the dedupe key (exact for Han-containing words,
/// ASCII case-insensitive otherwise). Entries the user added or edited since
/// learning stay untouched unless they match a learned word. Returns how many
/// entries were removed.
pub(crate) fn remove_learned_words(learned: &[String], custom_words: &mut Vec<String>) -> usize {
    let before = custom_words.len();
    custom_words.retain(|existing| {
        !learned
            .iter()
            .any(|word| same_word_in(word, std::slice::from_ref(existing)))
    });
    before - custom_words.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn learn(pasted: &str, field: &str) -> Vec<String> {
        extract_learned_words(pasted, field, &[])
    }

    #[test]
    fn unchanged_paste_learns_nothing() {
        assert!(learn("hello world", "so I said hello world and left").is_empty());
        assert!(learn("你好世界", "前面 你好世界 后面").is_empty());
    }

    #[test]
    fn learns_han_homophone_correction() {
        // 威信 and 微信 are exact toneless-pinyin homophones (wei xin).
        assert_eq!(
            learn("我在用威信聊天", "我在用微信聊天"),
            vec!["微信".to_string()]
        );
    }

    #[test]
    fn learns_han_near_pinyin_correction() {
        // 自持 (zi chi) vs 支持 (zhi chi): z/zh initial confusion.
        assert_eq!(learn("请自持一下", "请支持一下"), vec!["支持".to_string()]);
    }

    #[test]
    fn rejects_unrelated_han_correction() {
        // 苹果 (ping guo) vs 香蕉 (xiang jiao): no pinyin overlap.
        assert!(learn("我在吃苹果", "我在吃香蕉").is_empty());
    }

    #[test]
    fn rejects_han_correction_with_wrong_syllable_in_short_word() {
        // 微小 (wei xiao) vs 微信 (wei xin): one of two syllables wrong.
        assert!(learn("这个想法很微小", "这个想法很微信").is_empty());
    }

    #[test]
    fn rejects_han_word_longer_than_eight_chars() {
        let old = "一二三四五六七八九";
        let new = "一二三四五六七八零";
        // All syllables differ; also nine characters long.
        assert!(learn(old, new).is_empty());
    }

    #[test]
    fn learns_latin_fuzzy_correction() {
        assert_eq!(
            learn("use handee notes today", "use handy notes today"),
            vec!["handy".to_string()]
        );
    }

    #[test]
    fn learns_latin_casing_of_term() {
        // Interior case change: a genuine spelling preference.
        assert_eq!(
            learn("i use chargebee a lot", "i use ChargeBee a lot"),
            vec!["ChargeBee".to_string()]
        );
    }

    #[test]
    fn learns_spoken_multiword_term() {
        // "charge bee" (spoken) corrected to the compound "ChargeBee".
        assert_eq!(
            learn("open charge bee settings", "open ChargeBee settings"),
            vec!["ChargeBee".to_string()]
        );
    }

    #[test]
    fn rejects_sentence_start_capitalization_only() {
        // "hello" -> "Hello" is indistinguishable from a sentence-start fix.
        assert!(learn("hello world", "Hello world").is_empty());
    }

    #[test]
    fn rejects_dissimilar_latin_correction() {
        assert!(learn("meeting at noon", "meeting at dusk").is_empty());
    }

    #[test]
    fn ignores_pure_insertions_and_deletions() {
        assert!(learn("hello world", "hello big world").is_empty());
        assert!(learn("hello big world", "hello world").is_empty());
    }
    #[test]
    fn rejects_large_rewrite() {
        // Six of ten words replaced: a rewrite, not a term fix.
        let pasted = "one two three four five six seven eight nine ten";
        let field = "one aaa bbb ccc ddd eee fff eight nine ten";
        assert!(learn(pasted, field).is_empty());
    }

    #[test]
    fn dedupes_against_existing_custom_words() {
        let existing = vec!["微信".to_string(), "Handy".to_string()];
        let learned = extract_learned_words("我在用威信聊天", "我在用微信聊天", &existing);
        assert!(learned.is_empty());

        // Latin dedupe is case-insensitive.
        let learned = extract_learned_words("use handee notes", "use Handy notes", &existing);
        assert!(learned.is_empty());
    }

    #[test]
    fn learns_correction_at_start_of_short_paste() {
        // The prefix anchor covers the correction; the suffix anchor and the
        // last word atom must still locate the region.
        assert_eq!(
            learn("威信 很好用", "微信 很好用"),
            vec!["微信".to_string()]
        );
    }

    #[test]
    fn learns_correction_at_end_of_paste() {
        assert_eq!(
            learn("打开我的威信", "打开我的微信"),
            vec!["微信".to_string()]
        );
    }

    #[test]
    fn learns_mid_paste_correction_with_surrounding_text() {
        let pasted = "我们等下发一条威信给你";
        let field = "之前的内容。我们等下发一条微信给你，之后的补充。";
        assert_eq!(learn(pasted, field), vec!["微信".to_string()]);
    }

    #[test]
    fn skips_when_paste_cannot_be_located() {
        // Field no longer contains any anchor of the paste (e.g. the message
        // was submitted and the field cleared).
        assert!(learn("hello world", "completely unrelated text").is_empty());
        assert!(learn("hello world", "").is_empty());
    }

    #[test]
    fn dedupes_repeated_learns_within_one_paste() {
        // Both spans corrected to the same term in one paste.
        let pasted = "用威信联系，或发威信消息";
        let field = "用微信联系，或发微信消息";
        assert_eq!(learn(pasted, field), vec!["微信".to_string()]);
    }

    #[test]
    fn rejects_mixed_script_replacement_but_keeps_han_term() {
        // The user inserted "GPT" after fixing 威信 -> 微信: the insertion is
        // ignored and the pure-Han term is still learned.
        assert_eq!(
            learn("打开威信看看", "打开微信GPT看看"),
            vec!["微信".to_string()]
        );
    }

    #[test]
    fn rejects_multi_word_latin_replacement() {
        // Corrected side must be a single term.
        assert!(learn("deploy the api key", "deploy the access key").is_empty());
    }

    #[test]
    fn atomization_is_lossless() {
        for text in [
            "hello world",
            "你好，世界！charge bee & co.",
            "  spaced\tout\n",
            "",
        ] {
            assert_eq!(atomize(text).concat(), text);
        }
    }

    #[test]
    fn atom_helpers_classify_correctly() {
        assert!(is_word_atom("hello"));
        assert!(is_word_atom("你"));
        assert!(!is_word_atom(" "));
        assert!(!is_word_atom("，"));
        assert_eq!(atom_core("chatgpt."), Some("chatgpt"));
        assert_eq!(atom_core("..."), None);
    }

    #[test]
    fn differs_only_by_leading_case_cases() {
        assert!(differs_only_by_leading_case("hello", "Hello"));
        assert!(!differs_only_by_leading_case("chargebee", "ChargeBee"));
        assert!(!differs_only_by_leading_case("abc", "abc"));
    }

    #[test]
    fn char_helpers_respect_boundaries() {
        let s = "a你bc"; // a(1) 你(3) b(1) c(1)
        assert_eq!(forward_chars(s, 0, 3), 5);
        assert_eq!(back_chars(s, 5, 2), 1);
        assert_eq!(floor_char_boundary(s, 2), 1);
    }

    #[test]
    fn remove_learned_words_removes_only_the_learned_batch() {
        // A real learn batch never contains words that already existed
        // (dedupe at learn time uses the same match rule).
        let mut words: Vec<String> = ["微信", "Handy", "chatgpt"]
            .iter()
            .map(|w| w.to_string())
            .collect();
        let learned = ["微信".to_string(), "handy".to_string()];

        let removed = remove_learned_words(&learned, &mut words);

        assert_eq!(removed, 2);
        assert_eq!(words, vec!["chatgpt".to_string()]);
    }

    #[test]
    fn remove_learned_words_keeps_unrelated_and_similar_words() {
        let mut words: Vec<String> = ["威信".to_string(), "ChatGPT".to_string()]
            .iter()
            .map(|w| w.to_string())
            .collect();
        let learned = ["微信".to_string()];

        let removed = remove_learned_words(&learned, &mut words);

        assert_eq!(removed, 0);
        assert_eq!(words.len(), 2);
    }

    #[test]
    fn remove_learned_words_is_a_noop_when_already_deleted() {
        let mut words: Vec<String> = vec!["别的".to_string()];
        let learned = ["微信".to_string()];

        assert_eq!(remove_learned_words(&learned, &mut words), 0);
        assert_eq!(words, vec!["别的".to_string()]);
    }
}
