use std::iter;

pub fn unique_char_ngrams(s: &str, min_n: usize, max_n: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    if !chars.is_empty() {
        let mut v = Vec::<String>::from_iter(
            (min_n.min(chars.len())..=max_n.min(chars.len()))
                .flat_map(|n| chars.windows(n).map(|t| t.iter().collect())),
        );
        v.sort_unstable();
        v.dedup();
        v
    } else {
        Vec::new()
    }
}

pub fn derepeat(text: &str, n: usize) -> String {
    let mut last_char: char = '𝕊';
    let mut repeat: usize = 0;
    text.chars()
        .filter(|c| {
            if last_char == *c {
                repeat += 1;
            } else {
                repeat = 0;
                last_char = *c;
            }
            repeat < n
        })
        .collect()
}

pub fn space_around_ic(text: &str) -> String {
    let mut last_char: char = '𝕊';
    let mut repeat: u32 = 0;
    text.chars()
        .flat_map(|c| {
            let next_chars = if c != 'ㅋ'
                || c != 'ㅎ'
                || c != 'ㅜ'
                || c != 'ㅠ'
                || c != 'ㄷ'
                || c != '!'
                || c != '.'
                || c != ','
                || c != '?'
            {
                if last_char == 'ㅋ'
                    || last_char == 'ㅎ'
                    || last_char == 'ㅜ'
                    || last_char == 'ㅠ'
                    || last_char == 'ㄷ'
                    || last_char == '!'
                    || last_char == '.'
                    || last_char == ','
                    || last_char == '?'
                {
                    repeat += 1;
                } else {
                    repeat = 0;
                }
                iter::once(c).chain(iter::once('\0'))
            } else if repeat >= 1 {
                repeat = 0;
                iter::once(c).chain(iter::once(' '))
            } else {
                iter::once(c).chain(iter::once('\0'))
            };
            last_char = c;
            next_chars
        })
        .filter(|c| c != &'\0')
        .collect()
}

pub fn normalize(text: &str) -> String {
    space_around_ic(&derepeat(text, 3))
}

pub fn tokenize(text: &str, ngram: usize) -> Vec<String> {
    let tokens: Vec<_> = text.split(char::is_whitespace).collect();
    if ngram > 1 {
        tokens.windows(ngram).map(|t| t.join(" ")).collect()
    } else {
        tokens.into_iter().map(|t| t.to_owned()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_tokenize() {
        let text = "하나 둘 셋 넷";
        let expect = tokenize(text, 3);
        assert_eq!(expect, vec!["하나 둘 셋", "둘 셋 넷"]);
        let expect = tokenize(text, 2);
        assert_eq!(expect, vec!["하나 둘", "둘 셋", "셋 넷"]);
        let expect = tokenize(text, 1);
        assert_eq!(expect, vec!["하나", "둘", "셋", "넷"]);
    }
}
