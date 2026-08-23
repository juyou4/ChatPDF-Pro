/**
 * 关闭指定 micromark construct。移植自 Cherry Studio，
 * 用来关掉缩进代码块，避免公式行被当成 code。
 */
function add(data, field, value) {
  const list = data[field] ? data[field] : (data[field] = []);
  list.push(value);
}

export default function remarkDisableConstructs(constructs = []) {
  return function remarkDisableConstructsPlugin() {
    if (!constructs.length) return;
    add(this.data(), 'micromarkExtensions', {
      disable: { null: constructs },
    });
  };
}
