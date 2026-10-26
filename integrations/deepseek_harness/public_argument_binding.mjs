/** Literal argument projections from original public receipts. No evaluation or inference. */
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const FIELD = /^(?:[A-Za-z_][A-Za-z0-9_]*|\$scalar|\$single)$/;
export function bindingSpec(value) {
  return object(value) && Object.keys(value).join() === '$public'
    && object(value.$public)
    && Object.keys(value.$public).sort().join() === 'after,field'
    && /^(?:root|[A-P])$/.test(value.$public.after)
    && typeof value.$public.field === 'string' && FIELD.test(value.$public.field);
}

// Tokenize strings before looking for fields: content="ride_id='fake'" is data.
// The scanner accepts only native quoted strings, identifiers and punctuation.
function tokens(text) {
  const result = [];
  for (let i = 0; i < text.length;) {
    const ch = text[i];
    if (/\s/.test(ch)) { i++; continue; }
    if (ch === '"' || ch === "'") {
      const quote = ch; let value = ''; let closed = false; i++;
      while (i < text.length) {
        let next = text[i++];
        if (next === quote) { closed = true; break; }
        if (next === '\\') {
          if (i >= text.length) return null;
          next = text[i++];
          const escape = { '\\': '\\', '"': '"', "'": "'", n: '\n', r: '\r', t: '\t',
            b: '\b', f: '\f', a: '\x07', v: '\x0b' };
          if (Object.hasOwn(escape, next)) value += escape[next];
          else if (['x', 'u', 'U'].includes(next)) {
            const size = { x: 2, u: 4, U: 8 }[next];
            const digits = text.slice(i, i + size);
            if (digits.length !== size || !/^[0-9a-fA-F]+$/.test(digits)) return null;
            const code = Number.parseInt(digits, 16);
            if (code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) return null;
            value += String.fromCodePoint(code); i += size;
          } else return null;
        } else {
          if (next === '\n' || next === '\r') return null;
          value += next;
        }
      }
      if (!closed || value.includes('\0')) return null;
      result.push({ kind: 'string', value });
    } else if (/[A-Za-z_]/.test(ch)) {
      const begin = i++;
      while (i < text.length && /[A-Za-z0-9_]/.test(text[i])) i++;
      result.push({ kind: 'identifier', value: text.slice(begin, i) });
    } else { result.push({ kind: 'punctuation', value: ch }); i++; }
  }
  return result;
}

export function uniquePublicString(text, field) {
  if (typeof text !== 'string' || typeof field !== 'string' || !FIELD.test(field)) return null;
  if (field === '$scalar') {
    const value = text.trim();
    return /^[A-Za-z0-9_+@.:/-]+$/.test(value) ? value : null;
  }
  const stream = tokens(text);
  if (!stream) return null;
  if (field === '$single') {
    return stream.length === 3 && stream[0].value === '[' && stream[2].value === ']'
      && stream[1].kind === 'string' ? stream[1].value : null;
  }
  const matches = [];
  for (let i = 0; i + 2 < stream.length; i++) {
    if (['identifier', 'string'].includes(stream[i].kind) && stream[i].value === field
        && [':', '='].includes(stream[i + 1].value)) {
      // A duplicated or non-string field makes the projection ambiguous.
      if (stream[i + 2].kind !== 'string') return null;
      matches.push(stream[i + 2].value);
    }
  }
  return matches.length === 1 ? matches[0] : null;
}

export function resolvePublicArguments(value, receipts, depth = 0) {
  if (depth > 16) throw new Error('Public argument template is too deep');
  if (object(value) && Object.hasOwn(value, '$public')) {
    if (!bindingSpec(value)) throw new Error('Invalid closed public argument binding');
    const { after, field } = value.$public;
    const receipt = receipts.get(after);
    if (!receipt || receipt.isError === true || !Array.isArray(receipt.content)
        || receipt.content.length !== 1 || receipt.content[0].type !== 'text') {
      throw new Error('Public argument binding lacks its actual successful receipt');
    }
    const resolved = uniquePublicString(receipt.content[0].text, field);
    if (resolved === null) throw new Error('Public argument binding is missing or ambiguous');
    return resolved;
  }
  if (Array.isArray(value)) return value.map(item => resolvePublicArguments(item, receipts, depth + 1));
  if (object(value)) return Object.fromEntries(Object.entries(value).map(
    ([key, item]) => [key, resolvePublicArguments(item, receipts, depth + 1)],
  ));
  return value;
}
