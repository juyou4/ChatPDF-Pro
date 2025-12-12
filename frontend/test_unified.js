import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';

const content = `
# Math Test

Inline: $E=mc^2$

Display:
$$
\\frac{1}{2}
$$
`;

async function main() {
    try {
        const file = await unified()
            .use(remarkParse)
            .use(remarkMath)
            .use(remarkRehype)
            .use(rehypeKatex)
            .use(rehypeStringify)
            .process(content);

        console.log(String(file));
    } catch (error) {
        console.error('Error:', error);
    }
}

main();
