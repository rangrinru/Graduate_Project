import type { HangulBuffer } from "./types";

export const EMPTY_HANGUL_BUFFER: HangulBuffer = {
  cho: null,
  jung: null,
  jong: null,
};

const CHO_LIST = [
  "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
  "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
];

const JUNG_LIST = [
  "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
  "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
];

export const JONG_LIST = [
  "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
  "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
  "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
];

export const KOREAN_CONSONANTS = new Set([
  "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
  "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]);

export const KOREAN_VOWELS = new Set([
  "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅛ",
  "ㅜ", "ㅠ", "ㅡ", "ㅣ",
]);

export const DOUBLE_CHO_MAP: Record<string, string> = {
  "ㄱㄱ": "ㄲ",
  "ㄷㄷ": "ㄸ",
  "ㅂㅂ": "ㅃ",
  "ㅅㅅ": "ㅆ",
  "ㅈㅈ": "ㅉ",
};

export const COMPOUND_JUNG_MAP: Record<string, string> = {
  "ㅗㅏ": "ㅘ",
  "ㅗㅐ": "ㅙ",
  "ㅗㅣ": "ㅚ",
  "ㅜㅓ": "ㅝ",
  "ㅜㅔ": "ㅞ",
  "ㅜㅣ": "ㅟ",
  "ㅡㅣ": "ㅢ",
};

export const COMPOUND_JONG_MAP: Record<string, string> = {
  "ㄱㅅ": "ㄳ",
  "ㄴㅈ": "ㄵ",
  "ㄴㅎ": "ㄶ",
  "ㄹㄱ": "ㄺ",
  "ㄹㅁ": "ㄻ",
  "ㄹㅂ": "ㄼ",
  "ㄹㅅ": "ㄽ",
  "ㄹㅌ": "ㄾ",
  "ㄹㅍ": "ㄿ",
  "ㄹㅎ": "ㅀ",
  "ㅂㅅ": "ㅄ",
};

export const SPLIT_JONG_MAP: Record<string, [string, string]> = {
  "ㄳ": ["ㄱ", "ㅅ"],
  "ㄵ": ["ㄴ", "ㅈ"],
  "ㄶ": ["ㄴ", "ㅎ"],
  "ㄺ": ["ㄹ", "ㄱ"],
  "ㄻ": ["ㄹ", "ㅁ"],
  "ㄼ": ["ㄹ", "ㅂ"],
  "ㄽ": ["ㄹ", "ㅅ"],
  "ㄾ": ["ㄹ", "ㅌ"],
  "ㄿ": ["ㄹ", "ㅍ"],
  "ㅀ": ["ㄹ", "ㅎ"],
  "ㅄ": ["ㅂ", "ㅅ"],
};

export const KOREAN_KEY_ROWS = [
  ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅎ"],
  ["ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ"],
  ["ㅏ", "ㅑ", "ㅓ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ"],
  ["ㅐ", "ㅒ", "ㅔ", "ㅖ"],
];

export const ENGLISH_KEY_ROWS = [
  ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
  ["a", "s", "d", "f", "g", "h", "j", "k", "l"],
  ["z", "x", "c", "v", "b", "n", "m"],
];

export const NUMBER_KEY_ROWS = [
  ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
  ["-", "_", ".", "(", ")", "/", "@"],
];

export const composeHangul = (buffer: HangulBuffer) => {
  if (!buffer.cho && !buffer.jung && !buffer.jong) {
    return "";
  }

  if (buffer.cho && !buffer.jung) {
    return buffer.cho;
  }

  if (!buffer.cho && buffer.jung) {
    return buffer.jung;
  }

  if (!buffer.cho || !buffer.jung) {
    return "";
  }

  const choIndex = CHO_LIST.indexOf(buffer.cho);
  const jungIndex = JUNG_LIST.indexOf(buffer.jung);
  const jongIndex = buffer.jong ? JONG_LIST.indexOf(buffer.jong) : 0;

  if (choIndex < 0 || jungIndex < 0 || jongIndex < 0) {
    return `${buffer.cho}${buffer.jung}${buffer.jong || ""}`;
  }

  const unicode = 0xac00 + choIndex * 588 + jungIndex * 28 + jongIndex;

  return String.fromCharCode(unicode);
};

export const composeHangulWithoutJong = (buffer: HangulBuffer) => {
  return composeHangul({
    cho: buffer.cho,
    jung: buffer.jung,
    jong: null,
  });
};
