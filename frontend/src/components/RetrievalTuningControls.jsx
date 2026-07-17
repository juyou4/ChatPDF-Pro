import React from 'react';
import SettingsSegmentedControl from './SettingsSegmentedControl';

/**
 * 三态开关：null=自动（跟随后端 config）/ true=强制开 / false=强制关
 * 用于检索增强调优面板，让用户无需 restart 后端即可会话级切换 feature flag。
 *
 * 独立成小文件而不是放进 GlobalSettings.jsx：GlobalSettings 是 lazy() 懒加载面板，
 * 而这两个控件同时被设置中心「检索」tab（非懒加载）静态引用；混在一起会让整个
 * GlobalSettings.jsx 被打进主 bundle，失去代码分割的意义。
 */
export const TriStateToggle = ({ title, desc, value, onChange }) => {
    const options = [
        { value: null, label: '自动' },
        { value: true, label: '开' },
        { value: false, label: '关' },
    ];
    return (
        <div className="flex items-start justify-between gap-4 py-2">
            <div className="min-w-0 flex-1">
                <div className="text-[13px] font-bold text-gray-800">{title}</div>
                {desc && <div className="text-[11px] text-gray-500 mt-0.5 leading-relaxed">{desc}</div>}
            </div>
            <SettingsSegmentedControl
                ariaLabel={`${title}切换`}
                value={value}
                onChange={onChange}
                options={options}
                className="w-[154px] flex-shrink-0 rounded-[12px]"
                buttonClassName="px-1 py-1 text-[11px] font-bold text-center rounded-[9px]"
                indicatorClassName="rounded-[9px]"
            />
        </div>
    );
};

export const VisualVerificationMode = ({ value, onChange }) => {
    const normalized = ['auto', 'off', 'always'].includes(value) ? value : 'auto';
    const options = [
        { value: 'auto', label: '自动' },
        { value: 'off', label: '关闭' },
        { value: 'always', label: '总是' },
    ];
    return (
        <div className="flex items-start justify-between gap-4 py-2">
            <div className="min-w-0 flex-1">
                <div className="text-[13px] font-bold text-gray-800">表格视觉校验</div>
                <div className="text-[11px] text-gray-500 mt-0.5 leading-relaxed">
                    数值表格证据有风险时裁剪原 PDF 表格并让视觉模型抽取单元格；会增加一次视觉调用
                </div>
            </div>
            <SettingsSegmentedControl
                ariaLabel="表格视觉校验切换"
                value={normalized}
                onChange={onChange}
                options={options}
                className="w-[154px] flex-shrink-0 rounded-[12px]"
                buttonClassName="px-1 py-1 text-[11px] font-bold text-center rounded-[9px]"
                indicatorClassName="rounded-[9px]"
            />
        </div>
    );
};
